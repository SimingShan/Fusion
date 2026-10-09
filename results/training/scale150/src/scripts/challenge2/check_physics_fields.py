"""Validate fusion.physics_fields on DIII-D TRAIN labels and inspect nominal MAST fields.

Steps
1. Green function versus direct Biot-Savart integration.
2. DIII-D TRAIN shots (first N files in sorted order, K frames each): with
   phi = -efit_psirz, fit per frame phi = a*phi_F + b*phi_plasma + c*phi_ECOILA + d
   on pixels outside the label LCFS (scorer extract_lcfs), for three plasma
   placements (envelope-centroid filament, true-axis filament, Gaussian blob).
   Reports coefficient distributions and offset-free R2 outside the LCFS and on
   the full grid, and calibrates the ECOILA effective turn count.
3. MAST public-test shots: nominal phi_ext at all efit_times with per-frame
   statistics, coil/plasma decomposition, midplane B_Z null and decay index.
4. Timing of table build and per-frame evaluation.

Label use: DIII-D TRAIN labels only. No MAST, demo, hidden or audit labels.
"""
from __future__ import annotations

import argparse
import json
import sys
import time as wall_time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCORER = ROOT / "starter" / "fusion_scoring"
if str(SCORER) not in sys.path:
    sys.path.insert(0, str(SCORER))

from fusion import physics_fields as pf  # noqa: E402
from derive import _points_in_polygon  # noqa: E402
from lcfs import extract_lcfs  # noqa: E402
from o_point import find_o_point  # noqa: E402


def quantiles(values) -> dict:
    values = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if len(values) == 0:
        return {"count": 0}
    q = np.quantile(values, [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
    return {"count": int(len(values)), "min": float(q[0]), "p10": float(q[1]), "p25": float(q[2]),
            "median": float(q[3]), "p75": float(q[4]), "p90": float(q[5]), "max": float(q[6]),
            "mean": float(values.mean()), "std": float(values.std())}


def r2_offset_free(target: np.ndarray, model: np.ndarray, region: np.ndarray) -> float:
    """1 - var(target - model) / var(target) over region: any constant gauge is ignored."""
    t, m = target[region], model[region]
    resid = t - m
    return float(1.0 - resid.var() / max(t.var(), 1e-30))


def lstsq_fit(columns: list[np.ndarray], target: np.ndarray, region: np.ndarray) -> np.ndarray:
    A = np.stack([c[region] for c in columns] + [np.ones(int(region.sum()))], axis=1)
    coef, *_ = np.linalg.lstsq(A, target[region], rcond=None)
    return coef


def green_check() -> dict:
    cases = [(1.5, 0.7, 1.0, 0.0), (0.9, -0.3, 1.7, 0.5), (2.4, 1.5, 0.72, 0.0), (0.1, 0.0, 0.13, 0.4)]
    out = []
    for R, Z, Rc, Zc in cases:
        analytic = float(pf.green_function(R, Z, Rc, Zc))
        numeric = pf.biot_savart_psi(R, Z, Rc, Zc)
        out.append({"R": R, "Z": Z, "Rc": Rc, "Zc": Zc, "analytic_Wb_per_rad_per_A": analytic,
                    "biot_savart": numeric, "relative_error": (analytic - numeric) / numeric})
    reference = 1.524145e-7
    out[0]["reference_from_analyst"] = reference
    out[0]["matches_reference_1e-6"] = bool(abs(out[0]["analytic_Wb_per_rad_per_A"] - reference) / reference < 1e-6)
    # Parallelogram / rotation sanity: skew must not move the centroid.
    r, z = pf.rectangle_filaments(2.3733, 1.1171, 0.188, 0.1692, 0.0, 108.06, n=(6, 6))
    r2, z2 = pf.rectangle_filaments(1.0041, 1.5169, 0.1392, 0.1194, 45.0, 0.0, n=(6, 6))
    return {"cases": out, "skew_centroid_error_m": float(abs(r.mean() - 2.3733) + abs(z.mean() - 1.1171)),
            "rotation_centroid_error_m": float(abs(r2.mean() - 1.0041) + abs(z2.mean() - 1.5169))}


def diii_d_validation(args) -> tuple[dict, dict]:
    files = sorted((ROOT / "data/raw/data/diii_d_train").glob("*.parquet"))[:args.n_shots]
    grid_R, grid_Z, envelope = pf.load_envelope("DIII-D")
    RR, ZZ = np.meshgrid(grid_R, grid_Z)
    centroid = pf.envelope_centroid("DIII-D")
    grid = (grid_R, grid_Z)
    tables = None
    plasma_centroid = pf.plasma_filament_table(grid, *centroid)
    t0 = wall_time.perf_counter()
    plasma_blob = pf.plasma_blob_table(grid, *centroid, args.sigma)
    blob_seconds = wall_time.perf_counter() - t0
    placements = ["centroid_filament", "true_axis_filament", f"blob_sigma{args.sigma:g}"]
    records = []
    skipped = []
    frames_skipped = 0
    for path in files:
        inputs = pf.load_shot_inputs(path, with_labels=True)
        row = inputs["row"]
        if not np.allclose(inputs["grid_R"], grid_R) or not np.allclose(inputs["grid_Z"], grid_Z):
            raise ValueError(f"{path.name}: grid differs from the scorer envelope grid")
        if tables is None:
            tables = pf.coil_response_table(row, n=args.n_filaments)
        names = sorted(tables)
        f_names = [k for k in names if k != "magnetics_ECOILA"]
        T = len(inputs["efit_times"])
        frames = np.unique(np.linspace(0, T - 1, args.frames_per_shot).round().astype(int))
        for t in frames:
            phi = inputs["phi_target"][t]
            Ip = float(inputs["Ip_kA"][t])
            sign = float(inputs["Ip_sign"])
            if not inputs["Ip_mask"][t] or abs(Ip) < args.min_ip_kA or np.sign(Ip) != sign:
                # Residual Ip time-axis errors leave ~0 kA at early EFIT frames: unfit-able.
                frames_skipped += 1
                skipped.append({"shot": path.name, "frame": int(t), "reason": "ip", "Ip_kA": Ip})
                continue
            contour = extract_lcfs(phi, grid_R, grid_Z, "DIII-D", envelope, envelope.astype(float), axis_sign=1.0)
            if contour is None:
                frames_skipped += 1
                skipped.append({"shot": path.name, "frame": int(t), "reason": "no_lcfs", "Ip_kA": Ip})
                continue
            inside = _points_in_polygon(RR.ravel(), ZZ.ravel(), contour).reshape(phi.shape)
            outside = ~inside
            full = np.ones_like(outside)
            Ra, Za, _, _, axis_ok = find_o_point(phi, grid_R, grid_Z, envelope)
            currents = {k: float(inputs["currents"][k][t]) for k in names}
            phi_F = sign * sum(tables[k] * currents[k] for k in f_names)
            I_E = currents["magnetics_ECOILA"]
            phi_E = sign * tables["magnetics_ECOILA"] * I_E  # nominal single turn
            plasma_tables = {"centroid_filament": plasma_centroid,
                             "true_axis_filament": pf.plasma_filament_table(grid, Ra, Za) if axis_ok else None,
                             placements[2]: plasma_blob}
            record = {"shot": path.name, "frame": int(t), "time_ms": float(inputs["efit_times"][t]), "Ip_kA": Ip,
                      "I_ECOILA_kA": I_E, "axis_R": float(Ra), "axis_Z": float(Za), "axis_ok": bool(axis_ok),
                      "outside_fraction": float(outside.mean()), "fits": {}}
            for name in placements:
                table = plasma_tables[name]
                if table is None:
                    continue
                phi_p = abs(Ip) * table
                a, b, c, d = lstsq_fit([phi_F, phi_p, phi_E], phi, outside)
                model = a * phi_F + b * phi_p + c * phi_E
                a2, b2, d2 = lstsq_fit([phi_F, phi_p], phi, outside)
                model2 = a2 * phi_F + b2 * phi_p
                ac, cc, dc = lstsq_fit([phi_F, phi_E], phi - phi_p, outside)
                nominal = phi_F + phi_p
                record["fits"][name] = {
                    "a": float(a), "b": float(b), "c": float(c), "d": float(d),
                    "R2_outside": r2_offset_free(phi, model, outside), "R2_full": r2_offset_free(phi, model, full),
                    "no_ecoil": {"a": float(a2), "b": float(b2), "d": float(d2),
                                 "R2_outside": r2_offset_free(phi, model2, outside),
                                 "R2_full": r2_offset_free(phi, model2, full),
                                 "ecoil_unit_offset_Wb": float(sign * I_E * tables["magnetics_ECOILA"].mean())},
                    "b_fixed_1": {"a": float(ac), "c": float(cc), "d": float(dc),
                                  "R2_outside": r2_offset_free(phi, ac * phi_F + phi_p + cc * phi_E, outside)},
                    "nominal_a1_b1": {"R2_outside": r2_offset_free(phi, nominal, outside),
                                      "R2_full": r2_offset_free(phi, nominal, full)},
                }
            records.append(record)
    summary = {"n_shots": len(files), "frames_used": len(records), "frames_skipped": frames_skipped,
               "skipped": skipped, "min_ip_kA": args.min_ip_kA, "envelope_centroid": centroid, "sigma": args.sigma, "blob_table_seconds": blob_seconds,
               "placements": {}}
    for name in placements:
        fits = [r["fits"][name] for r in records if name in r["fits"]]
        if not fits:
            continue
        summary["placements"][name] = {
            "a": quantiles([f["a"] for f in fits]), "b": quantiles([f["b"] for f in fits]),
            "c_spatial_turns": quantiles([f["c"] for f in fits]), "d": quantiles([f["d"] for f in fits]),
            "R2_outside": quantiles([f["R2_outside"] for f in fits]),
            "R2_full": quantiles([f["R2_full"] for f in fits]),
            "no_ecoil_a": quantiles([f["no_ecoil"]["a"] for f in fits]),
            "no_ecoil_b": quantiles([f["no_ecoil"]["b"] for f in fits]),
            "no_ecoil_R2_outside": quantiles([f["no_ecoil"]["R2_outside"] for f in fits]),
            "no_ecoil_R2_full": quantiles([f["no_ecoil"]["R2_full"] for f in fits]),
            "b_fixed_1_a": quantiles([f["b_fixed_1"]["a"] for f in fits]),
            "b_fixed_1_c_turns": quantiles([f["b_fixed_1"]["c"] for f in fits]),
            "b_fixed_1_R2_outside": quantiles([f["b_fixed_1"]["R2_outside"] for f in fits]),
            "nominal_a1_b1_R2_outside": quantiles([f["nominal_a1_b1"]["R2_outside"] for f in fits]),
            "nominal_a1_b1_R2_full": quantiles([f["nominal_a1_b1"]["R2_full"] for f in fits]),
        }
    # ECOILA calibration by the offset method: the gauge offset of the no-ECOIL fit
    # (psi -> 0 at infinity physically) must be the near-uniform solenoid flux.
    calibration = {}
    if records:
        name = "centroid_filament"
        x = np.array([r["fits"][name]["no_ecoil"]["ecoil_unit_offset_Wb"] for r in records])
        y = np.array([r["fits"][name]["no_ecoil"]["d"] for r in records])
        slope, intercept = np.polyfit(x, y, 1)
        resid = y - (slope * x + intercept)
        per_shot = {}
        for shot in sorted(set(r["shot"] for r in records)):
            idx = [i for i, r in enumerate(records) if r["shot"] == shot]
            if len(idx) >= 3 and np.ptp(x[idx]) > 1e-6:
                per_shot[shot] = float(np.polyfit(x[idx], y[idx], 1)[0])
        ecoil_mean = float(tables["magnetics_ECOILA"].mean())
        ecoil_rel_variation = float(tables["magnetics_ECOILA"].std() / ecoil_mean)
        calibration = {
            "machine": "DIII-D", "label_role": "DIII-D TRAIN labels only (first files in sorted order)",
            "ecoila_effective_turns": float(slope),
            "ecoila_effective_turns_method": "regression of per-frame gauge offset d (fit without ECOILA term, "
                                             "centroid filament plasma) on sign(Ip)*I_ECOILA*mean(unit table)",
            "ecoila_offset_intercept_Wb_per_rad": float(intercept),
            "ecoila_offset_residual_rms_Wb_per_rad": float(resid.std()),
            "ecoila_offset_R2": float(1 - resid.var() / max(y.var(), 1e-30)),
            "ecoila_effective_turns_per_shot": per_shot,
            "ecoila_effective_turns_spatial_median": summary["placements"][name]["c_spatial_turns"]["median"],
            "ecoila_effective_turns_b_fixed_median": summary["placements"][name]["b_fixed_1_c_turns"]["median"],
            "ecoila_unit_table_mean_Wb_per_rad_per_kA_turn": ecoil_mean,
            "ecoila_unit_table_relative_spatial_std": ecoil_rel_variation,
            "ecoila_flux_per_kA_Wb_per_rad": float(slope * ecoil_mean),
            "fcoil_multiplier_a_median": summary["placements"][name]["b_fixed_1_a"]["median"],
            "plasma_multiplier_b_median": summary["placements"][name]["no_ecoil_b"]["median"],
            "n_frames": len(records), "n_shots": len(files), "shots": [p.name for p in files],
            "table_hash": pf.geometry_hash(row, args.n_filaments),
        }
    summary["records"] = records
    return summary, calibration


def mast_inspection(args) -> dict:
    files = sorted((ROOT / "data/raw/data/mast_public_test").glob("*.parquet"))[:args.mast_shots]
    grid_R, grid_Z, envelope = pf.load_envelope("MAST")
    centroid = pf.envelope_centroid("MAST")
    grid = (grid_R, grid_Z)
    plasma = pf.plasma_filament_table(grid, *centroid)
    blob = pf.plasma_blob_table(grid, *centroid, args.sigma)
    shots = []
    tables = None
    for path in files:
        inputs = pf.load_shot_inputs(path)
        row = inputs["row"]
        if tables is None:
            tables = pf.coil_response_table(row, n=args.n_filaments)
        groups = pf.coil_groups("MAST", sorted(tables))
        Ip = inputs["Ip_kA"]
        sign = inputs["Ip_sign"]
        phi = pf.external_field(tables, inputs["currents"], Ip, plasma, ip_sign=sign)
        phi_blob = pf.external_field(tables, inputs["currents"], Ip, blob, ip_sign=sign)
        parts = pf.decompose_external_field(tables, inputs["currents"], Ip, plasma, groups, ip_sign=sign)
        coil_only = parts["pcoils"] + parts["solenoid"]
        ir_c = int(np.argmin(np.abs(grid_R - centroid[0])))
        iz_c = int(np.argmin(np.abs(grid_Z - centroid[1])))
        frames = []
        for t in range(len(Ip)):
            stats = pf.frame_stats(phi[t], grid_R, grid_Z, centroid)
            blob_stats = pf.frame_stats(phi_blob[t], grid_R, grid_Z, centroid)
            midplane = pf.midplane_vertical_field(coil_only[t], grid_R, grid_Z)
            envelope_nulls = [r for r in midplane["null_R"] if grid_R[envelope[iz_c]].min() <= r <= grid_R[envelope[iz_c]].max()]
            frames.append({
                "frame": t, "time_ms": float(inputs["efit_times"][t]), "Ip_kA": float(Ip[t]),
                "I_sol_kA": float(inputs["currents"]["magnetics_sol_current"][t]),
                **stats,
                "blob_centre_value": blob_stats["centre_value"], "blob_peak_to_peak": blob_stats["peak_to_peak"],
                "blob_max": blob_stats["max"], "blob_min": blob_stats["min"], "Ip_sign": float(sign),
                "centre_pcoils": float(parts["pcoils"][t][iz_c, ir_c]),
                "centre_solenoid": float(parts["solenoid"][t][iz_c, ir_c]),
                "centre_plasma": float(parts["plasma"][t][iz_c, ir_c]),
                "mean_pcoils": float(parts["pcoils"][t].mean()),
                "mean_solenoid": float(parts["solenoid"][t].mean()),
                "mean_plasma": float(parts["plasma"][t].mean()),
                "rms_pcoils": float(np.sqrt((parts["pcoils"][t] ** 2).mean())),
                "rms_solenoid_centered": float(parts["solenoid"][t].std()),
                "rms_plasma_centered": float(parts["plasma"][t].std()),
                "coil_BZ_midplane_at_centroid_T": float(midplane["B_Z"][ir_c]),
                "coil_decay_index_at_centroid": float(midplane["decay_index"][ir_c]),
                "coil_BZ_null_R_all": midplane["null_R"],
                "coil_BZ_null_R_in_envelope": envelope_nulls,
                "solenoid_dominates_centre": bool(abs(parts["solenoid"][t][iz_c, ir_c]) >
                                                 abs(parts["pcoils"][t][iz_c, ir_c]) + parts["plasma"][t][iz_c, ir_c]),
                "solenoid_dominates_structure": bool(parts["solenoid"][t].std() > parts["plasma"][t].std()),
            })
        shots.append({"shot": path.name, "n_frames": len(frames), "frames": frames})
    all_frames = [f for s in shots for f in s["frames"]]
    keys = ["mean", "min", "max", "boundary_mean", "centre_value", "peak_to_peak", "blob_centre_value", "blob_peak_to_peak",
            "blob_max", "blob_min", "centre_pcoils", "centre_solenoid",
            "centre_plasma", "mean_solenoid", "mean_plasma", "rms_solenoid_centered", "rms_plasma_centered",
            "coil_BZ_midplane_at_centroid_T", "coil_decay_index_at_centroid", "I_sol_kA", "Ip_kA"]
    summary = {"shots": [s["shot"] for s in shots], "n_frames": len(all_frames), "envelope_centroid": centroid,
               "stats": {k: quantiles([f[k] for f in all_frames]) for k in keys},
               "fraction_solenoid_dominates_centre": float(np.mean([f["solenoid_dominates_centre"] for f in all_frames])),
               "fraction_solenoid_dominates_structure": float(np.mean([f["solenoid_dominates_structure"] for f in all_frames])),
               "fraction_with_midplane_null_in_envelope": float(np.mean([len(f["coil_BZ_null_R_in_envelope"]) > 0 for f in all_frames])),
               "null_R_in_envelope": quantiles([f["coil_BZ_null_R_in_envelope"][0] for f in all_frames if f["coil_BZ_null_R_in_envelope"]]),
               "dataset_card_reference": {"MAST_median_frame_peak_to_peak_Wb_per_rad": 0.21,
                                          "typical_flux_claim_plus_0.05_Vs_per_rad": "not found in data/raw/README.md; compared anyway"},
               "per_shot": shots}
    return summary


def timing(args) -> dict:
    out = {}
    for machine, split in (("DIII-D", "diii_d_train"), ("MAST", "mast_public_test")):
        path = sorted((ROOT / "data/raw/data" / split).glob("*.parquet"))[0]
        t0 = wall_time.perf_counter()
        inputs = pf.load_shot_inputs(path)
        load_seconds = wall_time.perf_counter() - t0
        t0 = wall_time.perf_counter()
        tables = pf.coil_response_table(inputs["row"], n=args.n_filaments, cache_dir=None)
        build_seconds = wall_time.perf_counter() - t0
        t0 = wall_time.perf_counter()
        pf.coil_response_table(inputs["row"], n=args.n_filaments)
        cached_seconds = wall_time.perf_counter() - t0
        grid = (inputs["grid_R"], inputs["grid_Z"])
        centroid = pf.envelope_centroid(machine)
        plasma = pf.plasma_filament_table(grid, *centroid)
        evaluator = pf.FieldEvaluator.from_tables(tables)
        currents = evaluator.currents_from(inputs["currents"])
        T = len(inputs["Ip_kA"])
        reps = max(1, 500 // T)
        t0 = wall_time.perf_counter()
        for _ in range(reps):
            for t in range(T):
                evaluator.coil_field(currents[t])
        evaluator_ms = (wall_time.perf_counter() - t0) / (reps * T) * 1e3
        t0 = wall_time.perf_counter()
        for t in range(T):
            pf.external_field(tables, {k: v[t] for k, v in inputs["currents"].items()}, inputs["Ip_kA"][t], plasma)
        convenience_ms = (wall_time.perf_counter() - t0) / T * 1e3
        t0 = wall_time.perf_counter()
        pf.external_field(tables, inputs["currents"], inputs["Ip_kA"], plasma)
        batch_ms = (wall_time.perf_counter() - t0) / T * 1e3
        t0 = wall_time.perf_counter()
        for t in range(T):
            pf.thomson_splat(grid, inputs["chord_R"], inputs["chord_Z"], inputs["pe"][t], inputs["pe_mask"][t])
        splat_ms = (wall_time.perf_counter() - t0) / T * 1e3
        out[machine] = {"shot": path.name, "frames": T, "load_inputs_seconds": load_seconds,
                        "table_build_seconds_cold": build_seconds, "table_load_seconds_cached": cached_seconds,
                        "filaments_total": int(sum(len(pf.rectangle_filaments(*g, n=pf.filament_counts(g[2], g[3], *grid, n=args.n_filaments))[0])
                                                  for g in np.stack([inputs["row"][k] for k in ("coil_R", "coil_Z", "coil_width", "coil_height", "coil_angle1", "coil_angle2")]).T)),
                        "evaluator_ms_per_frame": evaluator_ms, "external_field_convenience_ms_per_frame": convenience_ms,
                        "external_field_batch_ms_per_frame": batch_ms, "thomson_splat_ms_per_frame": splat_ms}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-shots", type=int, default=12)
    parser.add_argument("--frames-per-shot", type=int, default=6)
    parser.add_argument("--mast-shots", type=int, default=3)
    parser.add_argument("--sigma", type=float, default=0.25)
    parser.add_argument("--n-filaments", type=int, default=4)
    parser.add_argument("--min-ip-kA", type=float, default=100.0, help="skip DIII-D fit frames with |Ip| below this")
    parser.add_argument("--out", type=Path, default=ROOT / "results/challenge2/physics_fields_v1")
    parser.add_argument("--calibration-out", type=Path, default=pf.TABLE_DIR / "diii_d_calibration.json")
    parser.add_argument("--skip-mast", action="store_true")
    parser.add_argument("--skip-timing", action="store_true")
    parser.add_argument("--no-write-calibration", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    started = wall_time.perf_counter()
    report = {"green_check": green_check()}
    print("Green check:", json.dumps(report["green_check"]["cases"][0], indent=None))
    d3d, calibration = diii_d_validation(args)
    report["diii_d"] = {k: v for k, v in d3d.items() if k != "records"}
    print(f"DIII-D frames used {d3d['frames_used']} skipped {d3d['frames_skipped']}")
    for name, s in d3d["placements"].items():
        print(f"  {name}: a med {s['a']['median']:.3f} [{s['a']['p10']:.3f},{s['a']['p90']:.3f}] "
              f"b med {s['b']['median']:.3f} [{s['b']['p10']:.3f},{s['b']['p90']:.3f}] "
              f"c med {s['c_spatial_turns']['median']:.1f} d med {s['d']['median']:.4f} "
              f"R2out med {s['R2_outside']['median']:.4f} R2full med {s['R2_full']['median']:.4f} "
              f"noE R2out {s['no_ecoil_R2_outside']['median']:.4f} nominal R2out {s['nominal_a1_b1_R2_outside']['median']:.4f}")
    if calibration:
        print(f"ECOILA effective turns (offset regression) {calibration['ecoila_effective_turns']:.2f}, "
              f"R2 {calibration['ecoila_offset_R2']:.3f}, intercept {calibration['ecoila_offset_intercept_Wb_per_rad']:.4f}, "
              f"spatial median {calibration['ecoila_effective_turns_spatial_median']:.1f}, "
              f"b-fixed median {calibration['ecoila_effective_turns_b_fixed_median']:.1f}")
        if not args.no_write_calibration:
            args.calibration_out.parent.mkdir(parents=True, exist_ok=True)
            args.calibration_out.write_text(json.dumps(calibration, indent=2))
    report["calibration"] = calibration
    if not args.skip_mast:
        mast = mast_inspection(args)
        report["mast"] = {k: v for k, v in mast.items() if k != "per_shot"}
        s = mast["stats"]
        print(f"MAST {mast['n_frames']} frames: centre med {s['centre_value']['median']:.4f} p2p med {s['peak_to_peak']['median']:.4f} "
              f"blob centre med {s['blob_centre_value']['median']:.4f} blob p2p med {s['blob_peak_to_peak']['median']:.4f} "
              f"mean med {s['mean']['median']:.4f} boundary med {s['boundary_mean']['median']:.4f}; centre pcoils {s['centre_pcoils']['median']:.4f} "
              f"sol {s['centre_solenoid']['median']:.4f} plasma {s['centre_plasma']['median']:.4f}; sol dominates centre "
              f"{mast['fraction_solenoid_dominates_centre']:.2f} structure {mast['fraction_solenoid_dominates_structure']:.2f}; "
              f"null in envelope {mast['fraction_with_midplane_null_in_envelope']:.2f} decay index med {s['coil_decay_index_at_centroid']['median']:.3f}")
        (args.out / "mast_frames.json").write_text(json.dumps(mast["per_shot"], indent=1))
    if not args.skip_timing:
        report["timing"] = timing(args)
        for machine, t in report["timing"].items():
            print(f"timing {machine}: build {t['table_build_seconds_cold']:.2f}s ({t['filaments_total']} filaments), cached {t['table_load_seconds_cached']:.3f}s, "
                  f"evaluator {t['evaluator_ms_per_frame']:.4f} ms/frame, convenience {t['external_field_convenience_ms_per_frame']:.3f} ms, "
                  f"batch {t['external_field_batch_ms_per_frame']:.4f} ms, splat {t['thomson_splat_ms_per_frame']:.3f} ms")
    report["wall_seconds"] = wall_time.perf_counter() - started
    report["label_role"] = "DIII-D TRAIN labels only; no MAST/demo/hidden/audit labels"
    (args.out / "summary.json").write_text(json.dumps(report, indent=1))
    (args.out / "diii_d_frames.json").write_text(json.dumps(d3d["records"], indent=1))
    print(f"wrote {args.out / 'summary.json'} in {report['wall_seconds']:.1f}s")


if __name__ == "__main__":
    main()
