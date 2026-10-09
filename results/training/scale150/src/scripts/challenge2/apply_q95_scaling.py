#!/usr/bin/env python3
"""Replace ONLY the q95 channel of a zero-shot MAST prediction npz with the DIII-D-fitted physics
scaling (fit_q95_scaling.py), using the PREDICTED MAST flux geometry and the MAST inputs.

Per frame: LCFS from the predicted psi with the official scorer (MAST envelope, axis maximum,
axis_sign=+1) -> a, R0, kappa, tri_top, tri_bot (shot-median fallback when extraction fails);
B_T = mu0 * 24 * |I_TF| / (2 pi R0) with I_TF = magnetics_tf_current (kA -> A); Ip from the magnetics;
q95_hat from the selected model, clipped to [2, 12]. psirz and betaN are copied byte-identically.

Then validates the new npz with scripts/package_submission.validate_prediction (mast_public_test, no
packaging) and writes q95_scaling_report.json (coefficients, DIII-D R2s, MAST statistics, sha256).
Geometry + inputs are cached in <out>/mast_geometry_inputs.npz so a re-apply with another model is
instant (--model).
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "starter" / "fusion_scoring"))
sys.path.insert(0, str(ROOT / "scripts" / "challenge2"))
sys.path.insert(0, str(ROOT / "scripts"))
from fusion import c2_fields_data as cd  # noqa: E402
from fusion import physics_fields as pf  # noqa: E402
from q95_scaling_common import (GEOM_NAMES, MODEL_FORMULAS, Q95_CLIP, TF_TURNS, extract_geometry_frames,  # noqa: E402
                                fill_failed_geometry, predict_log_q95, q95_inputs, quantiles)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _shot_task(args):
    """Worker: geometry from the predicted MAST flux of one shot + its Ip / TF inputs."""
    index, parquet_path, source_npz = args
    with np.load(source_npz) as src:
        psi = src[f"shot_{index:04d}_psirz"].astype(np.float64)
    grid_R, grid_Z, env = pf.load_envelope("MAST")
    inputs = pf.load_shot_inputs(parquet_path, with_labels=False, thomson=False)
    assert inputs["machine"] == "MAST", parquet_path
    assert np.allclose(inputs["grid_R"], grid_R) and np.allclose(inputs["grid_Z"], grid_Z), parquet_path
    n = len(inputs["efit_times"])
    assert psi.shape == (n, 65, 65), (index, psi.shape, n)
    tf, tf_obs = cd.read_tf_current(Path(parquet_path), "MAST", inputs["efit_times"])
    geom = extract_geometry_frames(psi, grid_R, grid_Z, "MAST", env, axis_sign=1.0)
    return {"index": index, "geom": geom, "Ip_kA": inputs["Ip_kA"].astype(np.float64), "Ip_obs": inputs["Ip_mask"].astype(bool),
            "tf_kA": tf.astype(np.float64), "tf_obs": tf_obs.astype(bool), "times_ms": inputs["efit_times"].astype(np.float64)}


def build_cache(source_npz, raw_dir, cache_path, workers, max_shots=0):
    paths = cd.mast_parquet_paths(raw_dir)
    if max_shots:
        paths = paths[:max_shots]
    tasks = [(i, str(p), str(source_npz)) for i, p in enumerate(paths)]
    t0 = time.monotonic()
    with ProcessPoolExecutor(workers) as pool:
        results = list(pool.map(_shot_task, tasks, chunksize=4))
    results.sort(key=lambda r: r["index"])
    offsets = np.cumsum([0] + [len(r["Ip_kA"]) for r in results])
    cache = {"offsets": offsets, "geom": np.vstack([r["geom"] for r in results]),
             "Ip_kA": np.concatenate([r["Ip_kA"] for r in results]), "Ip_obs": np.concatenate([r["Ip_obs"] for r in results]),
             "tf_kA": np.concatenate([r["tf_kA"] for r in results]), "tf_obs": np.concatenate([r["tf_obs"] for r in results]),
             "times_ms": np.concatenate([r["times_ms"] for r in results]),
             "paths": np.array([str(p) for p in paths])}
    np.savez(cache_path, **cache)
    print(f"geometry cache: {len(paths)} shots, {offsets[-1]} frames, "
          f"{int((~np.isfinite(cache['geom']).all(axis=1)).sum())} LCFS failures, {time.monotonic() - t0:.0f}s -> {cache_path}", flush=True)
    return cache


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", type=Path, default=ROOT / "results/challenge2/c2_cnn_v3_main/mast/mast_public_test.npz")
    p.add_argument("--fit", type=Path, default=ROOT / "results/challenge2/c2_cnn_v3_main_q95fix/q95_scaling_fit.json")
    p.add_argument("--out", type=Path, default=ROOT / "results/challenge2/c2_cnn_v3_main_q95fix")
    p.add_argument("--raw-dir", type=Path, default=ROOT / "data/raw")
    p.add_argument("--model", default="pinned_p075",
                   help="model name from the fit json; only eps-PINNED models (pinned_*, wesson, iter89_shape) are "
                        "safe for MAST, the free-eps models extrapolate arbitrarily (see q95_scaling_common.py)")
    p.add_argument("--kappa-floor", type=float, default=0.0,
                   help="optional floor applied to the predicted MAST kappa before the scaling (0 = off); "
                        "the sensitivity table always reports the un-floored result")
    p.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    p.add_argument("--max-shots", type=int, default=0, help="smoke test only (skips validation)")
    p.add_argument("--rebuild-cache", action="store_true")
    p.add_argument("--npz-name", default="mast_public_test.npz")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    cache_path = args.out / ("mast_geometry_inputs.npz" if not args.max_shots else f"mast_geometry_inputs_smoke{args.max_shots}.npz")
    if cache_path.exists() and not args.rebuild_cache:
        with np.load(cache_path) as c:
            cache = {k: c[k] for k in c.files}
        print(f"loaded geometry cache {cache_path}", flush=True)
    else:
        cache = build_cache(args.source, args.raw_dir, cache_path, args.workers, args.max_shots)
    fit = json.loads(args.fit.read_text())
    model = fit["models"][args.model]
    coef = model["coefficients"]
    offsets = cache["offsets"]
    n_shots = len(offsets) - 1

    # ---- per-shot geometry fallback, then q95 ----
    geom_all = cache["geom"]
    finite = np.isfinite(geom_all).all(axis=1)
    global_median = np.median(geom_all[finite], axis=0)
    geom_filled = np.empty_like(geom_all)
    n_fail = 0
    shots_all_failed = 0
    for s in range(n_shots):
        sl = slice(offsets[s], offsets[s + 1])
        g, bad, nf = fill_failed_geometry(geom_all[sl], fallback=global_median)
        geom_filled[sl] = g
        n_fail += nf
        shots_all_failed += int(bad.all())
    Ip_A = np.abs(cache["Ip_kA"]) * 1e3
    tf_A = np.abs(cache["tf_kA"]) * 1e3
    tf_missing = ~cache["tf_obs"] | (tf_A < 1e3)
    if tf_missing.any():
        tf_A = np.where(tf_missing, np.median(tf_A[~tf_missing]), tf_A)
    ip_bad = ~cache["Ip_obs"] | (Ip_A < 5e4)
    geom_used = geom_filled.copy()
    kappa_floored = 0
    if args.kappa_floor > 0:
        kappa_floored = int((geom_used[:, 2] < args.kappa_floor).sum())
        geom_used[:, 2] = np.maximum(geom_used[:, 2], args.kappa_floor)
    feats = q95_inputs(geom_used, np.maximum(Ip_A, 5e4), tf_A, TF_TURNS["MAST"])
    feats_raw = q95_inputs(geom_filled, np.maximum(Ip_A, 5e4), tf_A, TF_TURNS["MAST"])

    # ---- sensitivity: every fitted model on the same MAST geometry/inputs (unclipped + clipped stats) ----
    sensitivity = {}
    for name, m in fit["models"].items():
        try:
            qr = np.exp(predict_log_q95(name, m["coefficients"], feats_raw))
        except (KeyError, ValueError):
            continue
        qc = np.clip(np.where(np.isfinite(qr), qr, Q95_CLIP[1]), *Q95_CLIP)
        sensitivity[name] = {"formula": MODEL_FORMULAS.get(name, fit["formulas"].get(name)),
                             "q95_unclipped_p05_med_p95": [float(v) for v in np.quantile(qr[np.isfinite(qr)], [.05, .5, .95])],
                             "q95_clipped_p05_med_p95": [float(v) for v in np.quantile(qc, [.05, .5, .95])],
                             "clipped_low_frac": float(np.mean(qr < Q95_CLIP[0])), "clipped_high_frac": float(np.mean(qr > Q95_CLIP[1])),
                             "median_in_band_3p5_7p5": bool(3.5 <= np.median(qc) <= 7.5),
                             "dev_r2_pred_geometry": m.get("dev_r2_pred_geometry")}
        print(f"  sensitivity {name:13s} MAST q95 p05/med/p95 = {sensitivity[name]['q95_clipped_p05_med_p95']} "
              f"clipped low/high {sensitivity[name]['clipped_low_frac']:.3f}/{sensitivity[name]['clipped_high_frac']:.3f}", flush=True)

    q95_raw = np.exp(predict_log_q95(args.model, coef, feats))
    q95 = np.clip(np.where(np.isfinite(q95_raw), q95_raw, Q95_CLIP[1]), *Q95_CLIP)
    for s in range(n_shots):   # frames with unusable Ip take the shot median of the usable frames
        sl = slice(offsets[s], offsets[s + 1])
        bad = ip_bad[sl]
        if bad.any() and (~bad).any():
            q95[sl][bad] = np.median(q95[sl][~bad])
    q95 = q95.astype(np.float32)

    # ---- smoothness: relative frame-to-frame change within each shot ----
    rel_steps, rel_steps_raw, shot_medians, shot_p05, shot_p95, jumps = [], [], [], [], [], 0
    for s in range(n_shots):
        q = q95[offsets[s]:offsets[s + 1]].astype(np.float64)
        qr = q95_raw[offsets[s]:offsets[s + 1]]
        shot_medians.append(np.median(q)); shot_p05.append(np.quantile(q, .05)); shot_p95.append(np.quantile(q, .95))
        if len(q) > 1:
            r = np.abs(np.diff(q)) / np.maximum(q[:-1], 1e-6)
            rel_steps.append(r); jumps += int((r > 0.2).sum())
            rel_steps_raw.append(np.abs(np.diff(qr)) / np.maximum(qr[:-1], 1e-6))
    rel_steps = np.concatenate(rel_steps) if rel_steps else np.zeros(0)
    rel_steps_raw = np.concatenate(rel_steps_raw) if rel_steps_raw else np.zeros(0)
    rel_steps_raw = rel_steps_raw[np.isfinite(rel_steps_raw)]
    stats = {"n_shots": n_shots, "n_frames": int(offsets[-1]), "q95": quantiles(q95),
             "q95_unclipped": quantiles(q95_raw), "clipped_low_frac": float(np.mean(q95_raw < Q95_CLIP[0])),
             "clipped_high_frac": float(np.mean(q95_raw > Q95_CLIP[1])),
             "per_shot_median_q95": quantiles(shot_medians), "per_shot_p05_q95": quantiles(shot_p05),
             "per_shot_p95_q95": quantiles(shot_p95),
             "smoothness": {"rel_step_median": float(np.median(rel_steps)) if len(rel_steps) else None,
                            "rel_step_p95": float(np.quantile(rel_steps, .95)) if len(rel_steps) else None,
                            "rel_step_p99": float(np.quantile(rel_steps, .99)) if len(rel_steps) else None,
                            "frac_steps_over_20pct": float(np.mean(rel_steps > 0.2)) if len(rel_steps) else None,
                            "n_steps_over_20pct": jumps,
                            "unclipped_rel_step_median": float(np.median(rel_steps_raw)) if len(rel_steps_raw) else None,
                            "unclipped_rel_step_p95": float(np.quantile(rel_steps_raw, .95)) if len(rel_steps_raw) else None,
                            "unclipped_rel_step_p99": float(np.quantile(rel_steps_raw, .99)) if len(rel_steps_raw) else None},
             "kappa_floor": args.kappa_floor, "kappa_floored_frames": kappa_floored,
             "sensitivity_all_models": sensitivity,
             "lcfs_failures_frames": int(n_fail), "lcfs_failure_frac": float(n_fail / offsets[-1]),
             "shots_all_frames_failed": shots_all_failed, "ip_unusable_frames": int(ip_bad.sum()),
             "tf_missing_frames": int(tf_missing.sum()),
             "features": {k: quantiles(v) for k, v in feats.items()},
             "geometry_global_median": dict(zip(GEOM_NAMES, [float(v) for v in global_median])),
             "in_expected_band_3p5_7p5_strict": bool(3.5 <= float(np.median(q95)) <= 7.5
                                                     and 3.5 <= float(np.quantile(q95, .05))
                                                     and float(np.quantile(q95, .95)) <= 7.5),
             "in_expected_band_3p5_7p5": bool(3.5 <= float(np.median(q95)) <= 7.5
                                              and 3.0 <= float(np.quantile(q95, .05))
                                              and float(np.quantile(q95, .95)) <= 9.0),
             "band_note": "strict = p05 >= 3.5 and p95 <= 7.5 and median in [3.5, 7.5]; "
                          "loose = median in [3.5, 7.5], p05 >= 3.0, p95 <= 9.0 ('roughly' in band)"}
    print(f"MAST q95 ({args.model}): median {stats['q95']['median']:.2f} p05 {stats['q95']['p05']:.2f} "
          f"p95 {stats['q95']['p95']:.2f}; rel-step median {stats['smoothness']['rel_step_median']}; "
          f"LCFS failures {n_fail}", flush=True)

    # ---- write npz: psirz and betaN byte-identical, q95 replaced ----
    out_npz = args.out / args.npz_name
    arrays = {}
    src_hashes = {}
    with np.load(args.source) as src:
        keys = src.files
        for k in keys:
            if k.endswith("_q95"):
                continue
            v = src[k]
            arrays[k] = v
            src_hashes[k] = hashlib.sha256(v.tobytes()).hexdigest() + f"|{v.dtype}|{v.shape}"
        for s in range(n_shots):
            key = f"shot_{s:04d}_q95"
            assert src[key].shape == (offsets[s + 1] - offsets[s],), (key, src[key].shape)
            arrays[key] = q95[offsets[s]:offsets[s + 1]]
    if args.max_shots:
        arrays = {k: v for k, v in arrays.items() if int(k.split("_")[1]) < args.max_shots}
        out_npz = args.out / f"smoke{args.max_shots}_{args.npz_name}"
    np.savez(out_npz, **arrays)
    with np.load(out_npz) as chk:
        for k, h in src_hashes.items():
            if k in arrays:
                v = chk[k]
                assert hashlib.sha256(v.tobytes()).hexdigest() + f"|{v.dtype}|{v.shape}" == h, f"{k} not byte-identical"
        assert chk["shot_0000_q95"].dtype == np.float32
    print(f"wrote {out_npz} ({out_npz.stat().st_size / 1e6:.1f} MB); psirz/betaN byte-identical to source", flush=True)

    validation = None
    if not args.max_shots:
        from package_submission import MACHINE_SHOTS, validate_prediction
        validation = validate_prediction(out_npz, args.raw_dir, expected_shots=MACHINE_SHOTS["mast_public_test"],
                                         config="mast_public_test")
        validation = {k: v for k, v in validation.items() if k != "shots"}
        print(f"validate_prediction: {validation['status']} ({validation['frames']} frames)", flush=True)

    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "script": "scripts/challenge2/apply_q95_scaling.py",
              "source_npz": str(args.source), "source_sha256": sha256_file(args.source),
              "output_npz": str(out_npz), "output_sha256": sha256_file(out_npz),
              "channels_replaced": ["q95"], "channels_byte_identical": ["psirz", "betaN"],
              "model": args.model, "coefficients": coef, "coefficient_order": model["coefficient_order"],
              "formula": MODEL_FORMULAS.get(args.model, fit["formulas"].get(args.model)),
              "model_choice_note": "eps dependence is not identifiable on DIII-D (eps 0.30-0.42); the chosen model pins "
                                   "the aspect-ratio factor to a physics form and fits only c0/kappa/delta on DIII-D. "
                                   "The MAST level therefore rests on that prior (see mast.sensitivity_all_models), "
                                   "not on data: true MAST q95 is unknown (no labels).", "q_cyl_formula": fit["formulas"]["q_cyl"],
              "q_iter89_formula": fit["formulas"]["q_iter89"], "q95_clip": list(Q95_CLIP),
              "tf_turns_mast": TF_TURNS["MAST"], "tf_signal": "magnetics_tf_current (kA)",
              "geometry_source": "official scorer extract_lcfs on PREDICTED MAST psi, MAST envelope, axis_sign=+1; "
                                 "shot-median fallback on failure",
              "diii_d": {"fit_json": str(args.fit), "train_shots": fit["train_shots_sampled"], "fit_frames": fit["fit_frames"],
                         "dev_frames": fit["dev"]["n_frames"],
                         "r2_dev_label_geometry": model["dev_r2_label_geometry"],
                         "r2_dev_pred_geometry": model["dev_r2_pred_geometry"],
                         "r2_dev_pred_geometry_clipped_2_12": model["dev_r2_pred_geometry_clipped_2_12"],
                         "r2_dev_network_head_same_frames": fit["dev"]["network_head_r2_same_frames"],
                         "r2_dev_network_head_val_score_json": fit["dev"]["network_head_r2_val_score_json"],
                         "train_r2_pooled": model["train_r2_pooled"],
                         "all_models_dev": fit["dev"]["models"],
                         "all_models_train_r2_log": {k: v["train_r2_log"] for k, v in fit["models"].items()},
                         "fit_notes": fit.get("notes"),
                         "all_models_coefficients": {k: v["coefficients"] for k, v in fit["models"].items()}},
              "mast": stats, "validation": validation, "elapsed_seconds": time.monotonic() - t0}
    rep_path = args.out / ("q95_scaling_report.json" if not args.max_shots else f"smoke{args.max_shots}_q95_scaling_report.json")
    rep_path.write_text(json.dumps(report, indent=1, default=float) + "\n")
    print(f"wrote {rep_path} in {time.monotonic() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
