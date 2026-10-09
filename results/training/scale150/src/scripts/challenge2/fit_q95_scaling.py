#!/usr/bin/env python3
"""Fit a physics-scaling estimate of q95 on DIII-D TRAINING labels (boundary geometry from the
official scorer's LCFS extraction on the LABEL flux, B_T from the TF coil current, Ip from the
magnetics) and evaluate it on the 128-shot dev fold with (a) label geometry and (b) the C2
network's predicted dev flux, pooled R2 as the scorer computes it.

Models (all ordinary least squares on log q95; <= 6 coefficients each):
    loglin6       log q95 = c0 + c1 log q_cyl + c2 log kappa + c3 eps + c4 eps^2 + c5 delta
    loglin_fixed  log q95 = log q_cyl + c0 + c2 log kappa + c3 eps + c5 delta   (q_cyl exponent 1, no eps^2)
    iter89        log q95 = c0 + c1 log q_iter89
    iter89_fixed  log q95 = c0 + log q_iter89
    pinned_p050/pinned_p075/pinned_p100
                  log q95 = log q_cyl - p log(1 - eps^2) + c0 + c2 log kappa + c5 delta   (p = 0.5 / 0.75 / 1)
    wesson        log q95 = log q_cyl + log(1 + eps^2) + c0 + c2 log kappa + c5 delta
    iter89_shape  log q95 = log q_cyl + log((1.17 - 0.65 eps)/(1 - eps^2)^2) + c0 + c2 log kappa + c5 delta
The DIII-D data span eps 0.30-0.42 only, so eps coefficients fitted freely (loglin6, loglin_fixed) are not
identifiable and extrapolate arbitrarily to MAST (eps ~ 0.77); the *pinned* models fix the aspect-ratio
factor to a physics form and fit only c0 / kappa / delta on DIII-D. They are indistinguishable on DIII-D.
where
    q_cyl    = 2 pi a^2 B_T (1+kappa^2)/2 / (mu0 R0 |Ip|)
    q_iter89 = 5 a^2 B_T / (R0 Ip[MA]) * (1 + kappa^2 (1 + 2 delta^2 - 1.2 delta^3))/2
               * (1.17 - 0.65 eps) / (1 - eps^2)^2                     (ITER-89 / Uckan shape form)
    B_T(R0)  = mu0 N_TF I_TF / (2 pi R0),  eps = a/R0,  delta = (tri_top + tri_bot)/2.

Inputs (never the audit split, never MAST labels): data/advanced_v1 train split (per-shot npz with
X raw features in kA, psi labels, scalars [q95, betaN]) and data/advanced_val_native_v3 val split
plus results/challenge2/c2_cnn_v3_main/val_predictions.npz (shot_%04d_psirz in manifest val order,
DIII-D convention: axis is a psi MINIMUM -> axis_sign=-1 as the scorer does).

Writes <out>/q95_scaling_fit.json with the coefficients, the fit-sample statistics and all dev R2s.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "starter" / "fusion_scoring"))
sys.path.insert(0, str(ROOT / "scripts" / "challenge2"))
from fusion import physics_fields as pf  # noqa: E402
from q95_scaling_common import (GEOM_NAMES, MODEL_FORMULAS, MODEL_NAMES, TF_TURNS, design_matrix,  # noqa: E402
                                extract_geometry_frames, fill_failed_geometry, pooled_r2, predict_log_q95,
                                q95_inputs, quantiles)

IP_INDEX, TF_INDEX, IP_OBS_INDEX, TF_OBS_INDEX = 20, 1, 41, 22


def _check_feature_names(manifest):
    names = manifest["feature_names"]
    assert names[IP_INDEX] == "magnetics_plasma_current", names[IP_INDEX]
    assert names[TF_INDEX] == "magnetics_bcoil", names[TF_INDEX]
    assert names[IP_OBS_INDEX] == "magnetics_plasma_current_observed", names[IP_OBS_INDEX]
    assert names[TF_OBS_INDEX] == "magnetics_bcoil_observed", names[TF_OBS_INDEX]
    assert manifest["scalar_names"] == ["q95", "betaN"]


def _shot_task(args):
    """Worker: geometry from the LABEL flux (and optionally a PREDICTED flux) of selected frames."""
    npz_path, frames, pred = args
    with np.load(npz_path) as d:
        X = d["X"][frames]
        psi = d["psi"][frames].astype(np.float64)
        scal = d["scalars"][frames]
        R, Z = d["R"].astype(np.float64), d["Z"].astype(np.float64)
    grid_R, grid_Z, env = pf.load_envelope("DIII-D")
    assert np.allclose(R, grid_R) and np.allclose(Z, grid_Z)
    out = {"frames": frames, "q95": scal[:, 0].astype(np.float64), "Ip_kA": X[:, IP_INDEX].astype(np.float64),
           "tf_kA": X[:, TF_INDEX].astype(np.float64), "Ip_obs": X[:, IP_OBS_INDEX] > 0.5, "tf_obs": X[:, TF_OBS_INDEX] > 0.5}
    out["geom_label"] = extract_geometry_frames(psi, grid_R, grid_Z, "DIII-D", env, axis_sign=-1.0)
    if pred is not None:
        out["geom_pred"] = extract_geometry_frames(np.asarray(pred, np.float64), grid_R, grid_Z, "DIII-D", env, axis_sign=-1.0)
    return out


def _collect(results, key, machine="DIII-D"):
    """Concatenate per-shot results into flat arrays; failed extractions filled with the shot median."""
    rows = []
    n_fail = 0
    for s, res in enumerate(results):
        geom, filled, fails = fill_failed_geometry(res[key])
        n_fail += fails
        rows.append(np.column_stack([geom, res["q95"], np.abs(res["Ip_kA"]) * 1e3, np.abs(res["tf_kA"]) * 1e3,
                                     res["Ip_obs"], res["tf_obs"], filled, np.full(len(filled), s)]))
    M = np.vstack(rows)
    n_geom = len(GEOM_NAMES)
    geom = M[:, :n_geom]
    q95, Ip_A, Itf_A, Ip_obs, tf_obs, filled, shot = (M[:, n_geom + j] for j in range(7))
    return geom, q95, Ip_A, Itf_A, Ip_obs > 0.5, tf_obs > 0.5, filled > 0.5, n_fail, shot.astype(int)


def _shot_median_fallback(q, bad, shot):
    """Frames with unusable inputs (Ip unobserved / < 100 kA) take the shot median of the usable frames."""
    q = np.array(q, float, copy=True)
    q = np.where(np.isfinite(q), q, np.nan)
    bad = bad | ~np.isfinite(q)
    for s in np.unique(shot):
        m = shot == s
        if (bad & m).any():
            q[bad & m] = np.nanmedian(q[m & ~bad]) if (m & ~bad).any() else np.nanmedian(q[~bad])
    return q


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train-dir", type=Path, default=ROOT / "data/advanced_v1")
    p.add_argument("--val-dir", type=Path, default=ROOT / "data/advanced_val_native_v3")
    p.add_argument("--val-predictions", type=Path, default=ROOT / "results/challenge2/c2_cnn_v3_main/val_predictions.npz")
    p.add_argument("--val-score", type=Path, default=ROOT / "results/challenge2/c2_cnn_v3_main/val_score.json")
    p.add_argument("--out", type=Path, default=ROOT / "results/challenge2/c2_cnn_v3_main_q95fix")
    p.add_argument("--n-shots", type=int, default=400)
    p.add_argument("--frames-per-shot", type=int, default=8)
    p.add_argument("--val-frames-per-shot", type=int, default=0, help="0 = every dev frame (scorer population)")
    p.add_argument("--seed", type=int, default=20260921)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--q95-range", type=float, nargs=2, default=(1.5, 15.0), help="label range kept for fitting")
    p.add_argument("--rebuild-cache", action="store_true", help="ignore <out>/diiid_geometry_cache.npz")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    cache_path = args.out / "diiid_geometry_cache.npz"
    cache = None
    if cache_path.exists() and not args.rebuild_cache:
        with np.load(cache_path, allow_pickle=True) as c:
            cache = {k: c[k] for k in c.files}
        same = (int(cache["seed"]) == args.seed and int(cache["n_shots"]) == args.n_shots
                and int(cache["frames_per_shot"]) == args.frames_per_shot
                and int(cache["val_frames_per_shot"]) == args.val_frames_per_shot)
        if not same:
            print("geometry cache settings differ from CLI -> recomputing", flush=True)
            cache = None
        else:
            print(f"loaded DIII-D geometry cache {cache_path}", flush=True)

    train_manifest = json.loads((args.train_dir / "manifest.json").read_text())
    val_manifest = json.loads((args.val_dir / "manifest.json").read_text())
    _check_feature_names(train_manifest)
    _check_feature_names(val_manifest)
    rng = np.random.default_rng(args.seed)
    train_entries = train_manifest["train"]
    chosen = sorted(rng.choice(len(train_entries), size=min(args.n_shots, len(train_entries)), replace=False).tolist())
    tasks = []
    for i in chosen:
        e = train_entries[i]
        n = int(e["n_frames"])
        frames = np.unique(np.linspace(0, n - 1, min(args.frames_per_shot, n)).round().astype(int))
        tasks.append((str(args.train_dir / e["path"]), frames, None))
    if cache is not None:
        train_results = list(cache["train_results"])
    else:
        with ProcessPoolExecutor(args.workers) as pool:
            train_results = list(pool.map(_shot_task, tasks, chunksize=8))
    geom, q95, Ip_A, Itf_A, Ip_obs, tf_obs, filled, n_fail, _ = _collect(train_results, "geom_label")
    print(f"train: {len(q95)} frames from {len(chosen)} shots, {n_fail} LCFS failures (shot-median filled), "
          f"{time.monotonic() - t0:.0f}s", flush=True)
    feats = q95_inputs(geom, Ip_A, Itf_A, TF_TURNS["DIII-D"])
    keep = (np.isfinite(q95) & (q95 >= args.q95_range[0]) & (q95 <= args.q95_range[1]) & Ip_obs & tf_obs
            & ~filled & np.isfinite(feats["q_cyl"]) & (Ip_A > 1e5) & (Itf_A > 1e4))
    print(f"fit sample: {int(keep.sum())} frames kept of {len(q95)}", flush=True)
    y = np.log(q95[keep])
    models = {}
    for name in MODEL_NAMES:
        A, offset, coef_names = design_matrix(name, {k: v[keep] for k, v in feats.items()})
        coef, *_ = np.linalg.lstsq(A, y - offset, rcond=None)
        yhat = A @ coef + offset
        models[name] = {"coefficients": dict(zip(coef_names, [float(c) for c in coef])),
                        "coefficient_order": coef_names,
                        "train_r2_log": float(1 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2)),
                        "train_r2_pooled": pooled_r2(q95[keep], np.exp(yhat)),
                        "train_rms_log": float(np.sqrt(np.mean((y - yhat) ** 2)))}
        print(f"  {name:13s} coef={models[name]['coefficients']} train pooled R2={models[name]['train_r2_pooled']:.4f}", flush=True)
    fit_stats = {k: quantiles(v[keep]) for k, v in feats.items()}
    fit_stats["q95_label"] = quantiles(q95[keep])
    fit_stats["ratio_q95_over_qcyl"] = quantiles(q95[keep] / feats["q_cyl"][keep])

    # ---- dev fold (128 shots): label geometry (upper bound) and predicted geometry (realistic) ----
    val_entries = val_manifest["val"]
    tasks = []
    with np.load(args.val_predictions) as vp:
        for i, e in enumerate(val_entries):
            n = int(e["n_frames"])
            pred = vp[f"shot_{i:04d}_psirz"]
            assert pred.shape[0] == n, (i, pred.shape, n)
            frames = (np.arange(n) if args.val_frames_per_shot <= 0
                      else np.unique(np.linspace(0, n - 1, min(args.val_frames_per_shot, n)).round().astype(int)))
            tasks.append((str(args.val_dir / e["path"]), frames, pred[frames]))
        head_q95 = np.concatenate([vp[f"shot_{i:04d}_q95"][t[1]] for i, t in enumerate(tasks)]).astype(np.float64)
    if cache is not None:
        val_results = list(cache["val_results"])
    else:
        with ProcessPoolExecutor(args.workers) as pool:
            val_results = list(pool.map(_shot_task, tasks, chunksize=2))
        np.savez(cache_path, train_results=np.array(train_results, dtype=object), val_results=np.array(val_results, dtype=object),
                 seed=args.seed, n_shots=args.n_shots, frames_per_shot=args.frames_per_shot,
                 val_frames_per_shot=args.val_frames_per_shot)
        print(f"saved DIII-D geometry cache -> {cache_path}", flush=True)
    geom_l, q95_v, Ip_v, Itf_v, Ipobs_v, tfobs_v, filled_l, fail_l, shot_v = _collect(val_results, "geom_label")
    geom_p, _, _, _, _, _, filled_p, fail_p, _ = _collect(val_results, "geom_pred")
    ip_bad_v = ~Ipobs_v | (Ip_v < 1e5)
    print(f"dev frames with unusable Ip (unobserved or < 100 kA): {int(ip_bad_v.sum())} -> shot-median q95 fallback", flush=True)
    print(f"dev: {len(q95_v)} frames, label LCFS failures {fail_l}, predicted LCFS failures {fail_p}, "
          f"{time.monotonic() - t0:.0f}s", flush=True)
    scored = np.isfinite(q95_v)   # the scorer masks nan labels only
    dev = {"n_frames": int(scored.sum()), "n_shots": len(val_entries), "frames_per_shot": args.val_frames_per_shot or "all",
           "lcfs_fail_label": int(fail_l), "lcfs_fail_pred": int(fail_p), "ip_unusable_frames": int(ip_bad_v.sum()),
           "q95_label": quantiles(q95_v[scored]),
           "network_head_r2_same_frames": pooled_r2(q95_v[scored], head_q95[scored]),
           "network_head_r2_val_score_json": json.loads(args.val_score.read_text())["r2_each"]["q95"]
           if args.val_score.exists() else None, "models": {}}
    feats_l = q95_inputs(geom_l, Ip_v, Itf_v, TF_TURNS["DIII-D"])
    feats_p = q95_inputs(geom_p, Ip_v, Itf_v, TF_TURNS["DIII-D"])
    for name, m in models.items():
        ql = np.exp(predict_log_q95(name, m["coefficients"], feats_l))
        qp = np.exp(predict_log_q95(name, m["coefficients"], feats_p))
        ql = _shot_median_fallback(ql, ip_bad_v, shot_v)
        qp = _shot_median_fallback(qp, ip_bad_v, shot_v)
        m["dev_r2_label_geometry"] = pooled_r2(q95_v[scored], ql[scored])
        m["dev_r2_pred_geometry"] = pooled_r2(q95_v[scored], qp[scored])
        m["dev_r2_pred_geometry_clipped_2_12"] = pooled_r2(q95_v[scored], np.clip(qp[scored], 2.0, 12.0))
        m["dev_q95_pred_geometry"] = quantiles(qp[scored])
        dev["models"][name] = {k: m[k] for k in ("dev_r2_label_geometry", "dev_r2_pred_geometry",
                                                 "dev_r2_pred_geometry_clipped_2_12")}
        print(f"  {name:13s} dev R2 label-geom={m['dev_r2_label_geometry']:.4f} pred-geom={m['dev_r2_pred_geometry']:.4f}", flush=True)
    print(f"  network q95 head R2 on same frames = {dev['network_head_r2_same_frames']:.4f}", flush=True)
    dev["geometry_pred_vs_label"] = {k: quantiles(geom_p[:, j] - geom_l[:, j]) for j, k in enumerate(GEOM_NAMES)}
    dev["features_label_geometry"] = {k: quantiles(v[scored]) for k, v in feats_l.items()}

    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "script": "scripts/challenge2/fit_q95_scaling.py",
              "seed": args.seed, "train_split": str(args.train_dir), "train_shots_sampled": len(chosen),
              "train_shot_indices": [int(train_entries[i]["shot_index"]) for i in chosen],
              "frames_per_shot": args.frames_per_shot, "fit_frames": int(keep.sum()), "train_lcfs_failures": int(n_fail),
              "q95_label_range_kept": list(args.q95_range), "tf_turns_diii_d": TF_TURNS["DIII-D"],
              "geometry_source": "official scorer extract_lcfs on LABEL psi, DIII-D envelope, axis_sign=-1",
              "formulas": {"q_cyl": "2*pi*a^2*B_T*(1+kappa^2)/2/(mu0*R0*|Ip|), B_T=mu0*N_TF*I_TF/(2*pi*R0)",
                           "q_iter89": "5*a^2*B_T/(R0*Ip_MA)*(1+kappa^2*(1+2*delta^2-1.2*delta^3))/2*(1.17-0.65*eps)/(1-eps^2)^2",
                           **MODEL_FORMULAS},
              "notes": ["The fit sample takes frames_per_shot linspace frames per shot including ramp-up/down phases "
                        "(q_cyl up to ~50, label q95 up to 15), so train_r2_pooled (linear space, pooled) is dominated by "
                        "that tail and is far below train_r2_log and the dev R2 (all dev frames, scorer population); "
                        "the OLS objective is log-space.",
                        "DIII-D eps spans 0.30-0.42 only: eps coefficients of loglin6/loglin_fixed are not identifiable "
                        "and must not be extrapolated to MAST; use a pinned_* / wesson / iter89_shape model there."],
              "fit_sample_statistics": fit_stats, "models": models, "dev": dev,
              "elapsed_seconds": time.monotonic() - t0}
    out = args.out / "q95_scaling_fit.json"
    out.write_text(json.dumps(report, indent=1, default=float) + "\n")
    print(f"wrote {out} in {time.monotonic() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
