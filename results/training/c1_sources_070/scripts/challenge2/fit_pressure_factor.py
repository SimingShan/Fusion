#!/usr/bin/env python3
"""Fit a total/electron pressure ratio using only permitted DIII-D training labels."""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import json
import sys

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from fusion.mast_profiles import read_row, FreeBoundaryGS, MU0
from fusion.mast_physics import _surface
from fusion.measured_pressure import measurements, PressureConstraint
from fusion.smooth_profiles import integral
from fusion.profile_scalars import vacuum_toroidal_function
from scripts.codabench_submit import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Use a fresh pressure-calibration directory")
    args.output.mkdir(parents=True)
    source = args.targets / "targets.npz"
    with np.load(source, allow_pickle=False) as z:
        data = dict(z)
    prior = joblib.load(args.prior)
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    if prior["training_source_sha256"] != source_hash or prior.get("closure_basis") != "edge_vanishing":
        raise ValueError("Require the matching frozen training source")
    for key in ("model", "coil_model"):
        prior[key].set_params(n_jobs=1)
    train = np.zeros(len(data["X"]), bool)
    train[prior["training_rows"]] = True
    if np.any(data["fold"][train] != "train"):
        raise ValueError("Pressure calibration includes held-out fitting rows")
    manifest = json.loads((args.data_root / "advanced_v1/manifest.json").read_text())
    entries = {int(e["shot_index"]): (fold, e) for fold in ("train", "val", "legacy_val") for e in manifest[fold]}
    predicted = prior["model"].predict(data["X"])*prior["scale"]+prior["center"]
    solver = FreeBoundaryGS(data["R"], data["Z"])
    factors = np.full(len(predicted), np.nan)
    raw_factors = np.full(len(predicted), np.nan)
    records, row, previous_shot = [], None, None
    for i in range(len(predicted)):
        shot = int(data["shot_index"][i]); frame = int(data["frame_index"][i])
        fold, entry = entries[shot]
        if fold != str(data["fold"][i]):
            raise ValueError("Training target identity differs from the permitted split")
        if shot != previous_shot:
            row = read_row(args.data_root / "raw" / entry["shard"], labels=True)
            previous_shot = shot
        record = {"row": i, "fold": fold, "shot_index": shot, "frame": frame, "valid": False}
        if not data["fit_valid"][i]:
            record["failure"] = "invalid_original_target"
            records.append(record)
            continue
        try:
            phi = data["phi"][i]
            np.testing.assert_allclose(phi, -row["efit_psirz"][frame], atol=1e-7, rtol=1e-7)
            surface = _surface(phi, solver, data["envelope"])
            if surface is None:
                raise ValueError("Target lacks a closed surface")
            contour, _, edge, delta, inside = surface
            time = float(row["efit_times"][frame])
            beta = float(row["efit_beta_n"][frame])
            Fvac, tf_observed = vacuum_toroidal_function(row, time)
            ip, length = float(data["Ip_A"][i]), float(data["L"][i])
            if not tf_observed or not np.isfinite(beta) or beta <= 0 or ip < 1e5:
                raise ValueError("Insufficient positive reference scalar/current observations")
            observations = measurements(row, time)
            if len(observations["pressure"]) < 8:
                raise ValueError("Too few measured pressure channels")
            minor = .5*np.ptp(contour[:, 0]); rgeom = .5*(contour[:, 0].max()+contour[:, 0].min())
            desired_mean = beta*(Fvac/rgeom)*(ip/1e6)/(200*MU0*minor)
            prior_axis = float(np.exp(predicted[i, 8])*MU0*ip**2/length**2)
            prior_shape = np.maximum(predicted[i, :4], 1e-8)
            s = np.clip((phi-edge)/delta, 0, 1)
            factor = 2.
            for iteration in range(4):
                constraint = PressureConstraint(observations, prior_axis, prior_shape, factor)
                axis, shape, info = constraint.fit(phi, solver, data["envelope"], surface)
                if not info.get("pressure_fit_used"):
                    raise ValueError("Reference pressure could not be fitted")
                electron = axis/factor*(integral(s)@shape)/(integral(1.)@shape)
                electron_mean = np.sum(electron[inside]*solver.RR[inside])/np.sum(solver.RR[inside])
                if not np.isfinite(electron_mean) or electron_mean < 100.:
                    raise ValueError("Electron pressure is too small for a stable ratio")
                implied = float(desired_mean/electron_mean)
                if not np.isfinite(implied) or implied <= 0:
                    raise ValueError("Nonpositive pressure ratio target")
                updated = float(np.clip(implied, 1., 20.))
                relative_change = abs(updated-factor)/factor
                factor = updated
                if relative_change < 1e-3:
                    break
            factors[i], raw_factors[i] = factor, implied
            record.update(valid=True, factor=factor, unconstrained_factor=implied,
                          iterations=iteration+1, pressure_relative_rmse=info["pressure_relative_rmse"],
                          relative_ratio_change=relative_change)
        except (ValueError, AssertionError) as exc:
            record["failure"] = f"{type(exc).__name__}: {exc}"
        records.append(record)
        if (i+1) % 128 == 0:
            print(json.dumps({"prepared_frames": i+1, "total_frames": len(predicted)}), flush=True)
    selected = train & np.isfinite(factors)
    if selected.sum() < 300 or len(np.unique(data["shot_index"][selected])) < 64:
        raise ValueError("Insufficient permitted pressure-ratio training coverage")
    _, inverse, counts = np.unique(data["shot_index"][selected], return_inverse=True, return_counts=True)
    weights = 1./counts[inverse]
    model = ExtraTreesRegressor(n_estimators=128, min_samples_leaf=8, max_depth=12,
                                max_features=.8, random_state=2073, n_jobs=1)
    model.fit(data["X"][selected], np.log(factors[selected]), sample_weight=weights)
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "training_frames": int(selected.sum()),
              "training_shots": len(counts), "target_source_sha256": source_hash,
              "parent_prior_sha256": hashlib.sha256(args.prior.read_bytes()).hexdigest(),
              "fitting_fold": "train", "audit_labels_read": False, "mast_labels_read": False,
              "input_features_unchanged": True, "ratio_bounds": [1., 20.],
              "training_targets_below_one": int(np.sum(raw_factors[selected] < 1.)),
              "training_targets_above_twenty": int(np.sum(raw_factors[selected] > 20.)),
              "training_factor_quantiles": np.quantile(factors[selected], [0., .25, .5, .75, 1.]).tolist(),
              "auxiliary_validation": {}, "eligible_for_submission": False,
              "limitations": "Training target geometry is used only to derive supervised ratio targets; no MAST target geometry or labels are available"}
    for fold in ("val", "legacy_val"):
        mask = (data["fold"] == fold) & np.isfinite(factors)
        estimate = model.predict(data["X"][mask])
        report["auxiliary_validation"][fold] = {"frames": int(mask.sum()),
            "constant_log_ratio_RMSE": float(np.sqrt(np.mean((np.log(2.)-np.log(factors[mask]))**2))),
            "predicted_log_ratio_RMSE": float(np.sqrt(np.mean((estimate-np.log(factors[mask]))**2))),
            "role": "Auxiliary target prediction only, not a reconstruction or official score"}
    prior["pressure_factor_model"] = model
    prior["pressure_factor_training"] = report
    joblib.dump(prior, args.output / "prior.joblib")
    write_json(args.output / "training.json", report)
    write_json(args.output / "target_records.json", {"records": records, "mast_labels_read": False, "audit_labels_read": False})
    np.savez(args.output / "ratio_targets.npz", factors=factors, unconstrained_factors=raw_factors, training_rows=np.flatnonzero(selected))
    (args.output / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
