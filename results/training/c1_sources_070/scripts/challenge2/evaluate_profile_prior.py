#!/usr/bin/env python3
"""Bounded input-only equilibrium reconstruction with a frozen closure prior."""
import argparse
from datetime import datetime, timezone
import json
import hashlib
from pathlib import Path
import sys

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from fusion.mast_profiles import read_row, input_features, learned_current, MU0, FreeBoundaryGS
from fusion.mast_physics import solve_profile_current, _surface
from fusion.mast_structure import check_separatrix
from fusion.profile_scalars import closure_scalars, vacuum_toroidal_function
from scripts.challenge2.prepare_profile_prior import clean
from scripts.codabench_submit import write_json


def infer(row, frame, prior, solver, envelope, previous=None, max_seconds=10, max_evaluations=600,
          measured_pressure=False, pressure_factor=2., observed_pressure_only=False, solver_method="krylov",
          constrain_pressure_current=False, positive_current_trials=False):
    X, ctx = input_features(row, frame)
    if ctx["feature_names"] != prior["feature_names"]:
        raise ValueError("Input feature contract changed")
    y = prior["model"].predict(X[None])[0] * prior["scale"] + prior["center"]
    ashape, cshape = np.maximum(y[:4], 1e-8), np.maximum(y[4:8], 1e-8)
    paxis = float(np.exp(y[8]) * ctx["pressure_scale"])
    ip, L = ctx["Ip_A"], ctx["L"]
    raxis, zaxis, width, elongation = y[9:] * np.array([L, L, L, 1])
    pressure_constraint = None
    if measured_pressure:
        if prior.get("closure_basis") != "edge_vanishing":
            raise ValueError("Measured pressure requires the matching edge basis")
        from fusion.measured_pressure import measurements, PressureConstraint
        if "pressure_factor_model" in prior:
            pressure_factor = float(np.exp(prior["pressure_factor_model"].predict(X[None])[0]))
            if not np.isfinite(pressure_factor) or not 1. <= pressure_factor <= 20.:
                raise ValueError("Invalid inferred total/electron pressure ratio")
        observations = measurements(row, ctx["time_ms"])
        if observed_pressure_only:
            observed = observations["observed"]
            observations = {"positions": observations["positions"][observed],
                "pressure": observations["pressure"][observed],
                "observed": observations["observed"][observed],
                "systems": [name for name, keep in zip(observations["systems"], observed) if keep]}
        pressure_constraint = PressureConstraint(observations, paxis, ashape, pressure_factor)
        if constrain_pressure_current:
            from fusion.current_constrained_pressure import CurrentConstrainedPressure
            pressure_constraint = CurrentConstrainedPressure(observations, paxis, ashape, pressure_factor, ip=ip)
    if row["source"] == "DIII-D":
        calibration = prior["coil_model"].predict(X[None])[0]
        vacuum = ctx["pf_flux"] + ctx["cs_unit"] * calibration[0] * ip
        gauge = calibration[1] * MU0 * ip * L
    else:
        vacuum = ctx["pf_flux"] + ctx["cs_nominal_flux"]
        gauge = 0.
        # DIII-D's vessel shape is not an admissible MAST inference prior. Use
        # actual measured pressure position and the published machine envelope.
        inside_coords = np.column_stack((solver.RR[envelope], solver.ZZ[envelope]))
        raxis, zaxis = np.median(inside_coords, axis=0)
        width = .38 * np.ptp(inside_coords[:, 0])
        elongation = np.ptp(inside_coords[:, 1]) / np.ptp(inside_coords[:, 0])
        if pressure_constraint is not None and len(observations["pressure"]) >= 4:
            weights = observations["pressure"]**2
            weights /= weights.sum()
            raxis, zaxis = weights@observations["positions"]
            support = observations["positions"][observations["pressure"] > .05*np.max(observations["pressure"]), 0]
            width = float(np.clip(np.ptp(support)/2, .25, .65))
    rho = ((solver.RR-raxis)/max(width,.05))**2 + ((solver.ZZ-zaxis)/max(width*elongation,.05))**2
    initial = np.maximum(1-rho, 0) * envelope if previous is None else np.maximum(previous, 0) * envelope
    initial.ravel()[solver.boundary] = 0
    if initial.sum() <= 0:
        raise ValueError("Predicted initialization has no plasma support")
    initial *= ip / (initial.sum()*solver.dR*solver.dZ)
    current_function, scalar_arguments = learned_current, {}
    if prior.get("closure_basis") == "edge_vanishing":
        from fusion.smooth_profiles import current, basis, integral
        current_function = current
        scalar_arguments = {"basis_function": basis, "integral_function": integral}
    def closure(phi):
        pressure, shape = paxis, ashape
        if pressure_constraint is not None:
            surface = _surface(phi, solver, envelope)
            if surface is None:
                raise ValueError("no_closed_surface_for_pressure_fit")
            pressure, shape, _ = pressure_constraint.fit(phi, solver, envelope, surface)
            return current_function(phi, solver, envelope, ip, pressure, shape, cshape, surface=surface)
        return current_function(phi, solver, envelope, ip, pressure, shape, cshape)
    current, details = solve_profile_current(vacuum, initial, solver, envelope, ip, paxis, 1.,
        method=solver_method, max_seconds=max_seconds, max_evaluations=max_evaluations, tolerance=1e-4,
        profile_function=closure, positive_trials=positive_current_trials)
    field = vacuum + solver.solve(current) + gauge
    surface = _surface(field, solver, envelope)
    if pressure_constraint is not None and surface is not None:
        paxis, ashape, diagnostics = pressure_constraint.fit(field, solver, envelope, surface)
        details.update(diagnostics)
    details.update(input_Ip_A=ip, input_only=True, pressure_axis_Pa=paxis,
                   pressure_factor_from_prior=measured_pressure and "pressure_factor_model" in prior,
                   observed_pressure_only=bool(observed_pressure_only),
                   Ip_relative_error=float(abs(current.sum()*solver.dR*solver.dZ-ip)/max(ip,1)),
                   current_min_A_per_m2=float(current.min()),
                   current_negative_mass_fraction=float(-np.minimum(current, 0).sum()/max(np.abs(current).sum(), 1e-30)),
                   linear_residual=solver.residual(solver.solve(current), current),
                   free_space_gauge_unverified=row["source"] == "MAST")
    if surface:
        contour, axis, edge, delta, _ = surface
        details["structure"] = check_separatrix(field, solver.R, solver.Z, contour, edge, delta, row)
        try:
            Fvac, observed = vacuum_toroidal_function(row, ctx["time_ms"])
            details.update(closure_scalars(field, solver, envelope, ip, paxis, ashape, cshape, Fvac, surface, **scalar_arguments))
            details["toroidal_current_observed"] = observed
        except ValueError as exc:
            details.update(scalar_valid=False, scalar_failure=str(exc))
    return field.astype(np.float32), current, details


def valid_prediction(result):
    field, _, info = result
    return (bool(info.get("scalar_valid")) and np.isfinite(field).all()
            and all(info.get(key) is not None and np.isfinite(info[key]) and info[key] > 0
                    for key in ("q95", "betaN")))


def infer_with_retry(row, frame, prior, solver, envelope, previous=None, max_seconds=10, max_evaluations=600,
                     measured_pressure=False, pressure_factor=2., observed_pressure_only=False, solver_method="krylov",
                     constrain_pressure_current=False, positive_current_trials=False):
    """Keep valid default outputs; retry invalid fits without extrapolated constraints.

    Validity depends only on predicted fields/scalars, never target values or
    shot identity. The fallback still predicts a complete field and derives both
    scalars from that field and closure; it never substitutes scalar values.
    """
    if observed_pressure_only or not measured_pressure:
        raise ValueError("The retry policy requires the default measured-pressure control")
    original, failure = None, None
    try:
        original = infer(row, frame, prior, solver, envelope, previous, max_seconds, max_evaluations,
                         measured_pressure, pressure_factor, False, solver_method, constrain_pressure_current, positive_current_trials)
        if valid_prediction(original):
            original[2]["pressure_observation_retry_used"] = False
            return original
        failure = original[2].get("scalar_failure", original[2].get("status", "invalid_default_prediction"))
    except (ValueError, FloatingPointError) as exc:
        failure = f"{type(exc).__name__}: {exc}"
    try:
        retried = infer(row, frame, prior, solver, envelope, previous, max_seconds, max_evaluations,
                        measured_pressure, pressure_factor, True, solver_method, constrain_pressure_current, positive_current_trials)
        retried[2].update(pressure_observation_retry_used=True, original_prediction_failure=failure)
        if valid_prediction(retried) or original is None:
            return retried
        original[2].update(pressure_observation_retry_used=False,
                           pressure_observation_retry_failure=retried[2].get("scalar_failure", retried[2].get("status")))
        return original
    except (ValueError, FloatingPointError) as exc:
        if original is None:
            raise ValueError(f"Both input-only attempts failed: {failure}; {type(exc).__name__}: {exc}") from exc
        original[2].update(pressure_observation_retry_used=False,
                           pressure_observation_retry_failure=f"{type(exc).__name__}: {exc}")
        return original


def infer_with_restart(row, frame, prior, solver, envelope, previous=None, max_seconds=10, max_evaluations=600,
                       measured_pressure=False, pressure_factor=2., observed_pressure_only=False, solver_method="krylov",
                       constrain_pressure_current=False, positive_current_trials=False):
    """Preserve every valid prediction; retry an invalid continuation from inputs.

    The first attempt is the existing complete conditional pressure policy.
    Only an invalid result with an actual previous state permits a new start.
    The new start uses the same measured inputs, closure, budgets and scalar
    derivation. No shot identifier, target, or replacement scalar is consulted.
    """
    options = dict(max_seconds=max_seconds, max_evaluations=max_evaluations,
        measured_pressure=measured_pressure, pressure_factor=pressure_factor,
        observed_pressure_only=observed_pressure_only, solver_method=solver_method,
        constrain_pressure_current=constrain_pressure_current, positive_current_trials=positive_current_trials)
    original, failure = None, None
    try:
        original = infer_with_retry(row, frame, prior, solver, envelope, previous, **options)
        if valid_prediction(original) or previous is None:
            return original
        failure = original[2].get("scalar_failure", original[2].get("status", "invalid_continuation"))
    except (ValueError, FloatingPointError) as exc:
        if previous is None:
            raise
        failure = f"{type(exc).__name__}: {exc}"
    try:
        retried = infer_with_retry(row, frame, prior, solver, envelope, None, **options)
        retried[2].update(initialization_retry_used=True, original_continuation_failure=failure)
        if valid_prediction(retried) or original is None:
            return retried
        original[2].update(initialization_retry_used=False,
                           initialization_retry_failure=retried[2].get("scalar_failure", retried[2].get("status")))
        return original
    except (ValueError, FloatingPointError) as exc:
        if original is None:
            raise ValueError(f"Both initializations failed: {failure}; {type(exc).__name__}: {exc}") from exc
        original[2].update(initialization_retry_used=False,
                           initialization_retry_failure=f"{type(exc).__name__}: {exc}")
        return original


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prior", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--shots", type=int, default=8)
    p.add_argument("--frames", type=int, default=3)
    p.add_argument("--max-seconds", type=float, default=10)
    p.add_argument("--max-evaluations", type=int, default=600)
    p.add_argument("--machines", nargs="+", choices=["DIII-D", "MAST"], default=["DIII-D", "MAST"])
    p.add_argument("--allow-approximate-seed", action="store_true")
    p.add_argument("--measured-pressure", action="store_true")
    p.add_argument("--pressure-factor", type=float, default=2.)
    p.add_argument("--observed-pressure-only", action="store_true")
    p.add_argument("--retry-observed-pressure", action="store_true")
    p.add_argument("--retry-fresh-initialization", action="store_true")
    p.add_argument("--solver-method", choices=("krylov", "anderson", "picard"), default="krylov")
    p.add_argument("--constrain-pressure-current", action="store_true")
    p.add_argument("--positive-current-trials", action="store_true")
    p.add_argument("--shard-count", type=int, default=1)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--prediction-jobs", type=int)
    p.add_argument("--data-root", type=Path, default=ROOT / "data")
    a = p.parse_args()
    if a.shard_count < 1 or not 0 <= a.shard_index < a.shard_count:
        p.error("Invalid shot-shard selection")
    if a.prediction_jobs is not None and a.prediction_jobs < 1:
        p.error("Prediction jobs must be positive")
    if a.retry_observed_pressure and (a.observed_pressure_only or not a.measured_pressure):
        p.error("Retry requires the default measured-pressure control")
    if a.retry_fresh_initialization and not a.retry_observed_pressure:
        p.error("Fresh initialization requires the existing conditional pressure policy")
    if a.constrain_pressure_current and not a.measured_pressure:
        p.error("A current-constrained pressure fit requires measured pressure")
    predictor = infer_with_retry if a.retry_observed_pressure else infer
    if a.retry_fresh_initialization:
        predictor = infer_with_restart
    a.output.mkdir(parents=True, exist_ok=True)
    if (a.output / "summary.json").exists():
        raise ValueError("Use a fresh evaluation directory")
    prior = joblib.load(a.prior)
    prior_hash = hashlib.sha256(a.prior.read_bytes()).hexdigest()
    if a.prediction_jobs is not None:
        for key in ("model", "coil_model"):
            if "n_jobs" in prior[key].get_params(deep=False):
                prior[key].set_params(n_jobs=a.prediction_jobs)
    code_files = [Path(__file__), ROOT / "fusion/mast_profiles.py", ROOT / "fusion/mast_physics.py", ROOT / "fusion/profile_scalars.py", ROOT / "fusion/smooth_profiles.py", ROOT / "fusion/measured_pressure.py"]
    if a.constrain_pressure_current:
        code_files.append(ROOT / "fusion/current_constrained_pressure.py")
    code_hashes = {str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest() for f in code_files}
    for source in code_files:
        destination = a.output / "source_snapshot" / source.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
    manifest = json.loads((a.data_root / "advanced_v1/manifest.json").read_text())
    inventory = json.loads((a.data_root / "raw/download_manifest.json").read_text())
    mast_paths = sorted(x["path"] for x in inventory["files"] if x["path"].startswith("data/mast_public_test/") and x["path"].endswith(".parquet"))
    selected = [("DIII-D", a.data_root / "raw" / manifest["val"][i]["shard"]) for i in
                np.unique(np.linspace(0, len(manifest["val"])-1, a.shots).astype(int))]
    selected += [("MAST", a.data_root / "raw" / mast_paths[i]) for i in
                 np.unique(np.linspace(0, len(mast_paths)-1, a.shots).astype(int))]
    selected = [(machine, path) for machine, path in selected if machine in a.machines]
    selected = selected[a.shard_index::a.shard_count]
    records = []
    for machine, path in selected:
        row = read_row(path, labels=False)
        solver = FreeBoundaryGS(row["efit_grid_R"], row["efit_grid_Z"])
        mask = "d3d" if machine == "DIII-D" else "mast"
        with np.load(ROOT / f"starter/fusion_scoring/masks/{mask}_envelope.npz") as z:
            envelope = z["mask_coarse"].astype(bool)
        previous = None
        seed_converged = None
        frames = (np.arange(len(row["efit_times"])) if a.frames == 0 else
                  np.unique(np.linspace(0, len(row["efit_times"])-1, a.frames).astype(int)))
        for frame in frames:
            record = {"machine": machine, "shot": path.stem, "frame": int(frame),
                      "continuation_used": previous is not None, "continuation_seed_converged": seed_converged}
            try:
                field, current, info = predictor(row, int(frame), prior, solver, envelope, previous, a.max_seconds,
                                           a.max_evaluations, a.measured_pressure, a.pressure_factor, a.observed_pressure_only,
                                           a.solver_method, a.constrain_pressure_current, a.positive_current_trials)
                record.update(info)
                if info["converged"]:
                    previous = current
                    seed_converged = True
                elif a.allow_approximate_seed and info.get("scalar_valid") and info.get("nonlinear_profile_relative_residual", 1) is not None:
                    if info["nonlinear_profile_relative_residual"] < .02 and np.min(current) >= -1e-6:
                        previous = current
                        seed_converged = False
                filename = f"{machine}_{path.stem}_{frame:04d}.npz"
                np.savez_compressed(a.output / filename, psi=field, current=current)
                # Open held-out DIII-D labels only after committing predictions.
                if machine == "DIII-D":
                    labels = read_row(path, labels=True)
                    target = -np.asarray(labels["efit_psirz"][frame], float)
                    record["development_flux_relative_rmse"] = float(np.linalg.norm(field-target)/max(np.linalg.norm(target),1e-12))
                    if info.get("scalar_valid"):
                        record["development_scalar_target"] = [float(labels["efit_q95"][frame]), float(labels["efit_beta_n"][frame])]
            except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
                record.update(converged=False, failure=str(exc))
            records.append(clean(record))
            print(json.dumps({"machine": machine, "frames_done": len(records), "converged": record.get("converged", False)}), flush=True)
            write_json(a.output / "progress.json", {"frames": len(records), "records": records})
    write_json(a.output / "summary.json", {"created_utc": datetime.now(timezone.utc).isoformat(),
        "records": records, "frames": len(records), "converged": sum(x.get("converged", False) for x in records),
        "settings": {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
        "mast_labels_read": False, "audit_labels_read": False, "official_score": False, "source_sha256": code_hashes,
        "prior_sha256": prior_hash,
        "eligible_for_submission": False,
        "limitations": "Bounded closure test; MAST gauge and closure accuracy require further validation."})


if __name__ == "__main__":
    main()
