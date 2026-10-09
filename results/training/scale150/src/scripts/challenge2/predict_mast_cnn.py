#!/usr/bin/env python3
"""Zero-shot MAST inference for the Challenge-2 flux U-Net plus an input-only plausibility report.

Loads every MAST public-test parquet (official order = lexicographically sorted pinned manifest
paths, as scripts/package_submission.py expects), builds the machine-agnostic channels with
fusion.c2_fields_data (MAST grid, MAST coil tables, sol/P-coil conventions, Thomson channel-0
drop), predicts all native frames and writes mast_public_test.npz with keys
shot_%04d_{psirz,q95,betaN}. MAST output psi = +phi_hat (axis maximum, AXIS_SIGN MAST = +1).
The checkpoint's stored feature spec (channels, field/scalar conditioning contract) must equal the
current fusion.c2_fields_data.feature_spec(); load_checkpoint refuses otherwise. The report also records
the MAST-side range of every conditioning entry actually fed to the model.

Checkpoints trained with synthetic MAST augmentation (train_c2_cnn.py --synthetic-dir) carry a ``synthetic`` entry that is
echoed in the report; their q95/betaN heads use the DIII-D z-scoring stored in the checkpoint like every other checkpoint,
so the inverse applied here is unchanged.

Report (no labels involved): per-shot LCFS extraction success with the official scorer on the
MAST envelope, R_axis/Z_axis/kappa/tri/volume/li distributions, fraction of Thomson chords with
p_e > 10% of the frame maximum lying outside the predicted LCFS, and per-frame psi statistics
(grid mean, axis value, boundary mean, peak-to-peak).
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "starter" / "fusion_scoring"))
from fusion import c2_fields_data as cd  # noqa: E402
from fusion import physics_fields as pf  # noqa: E402
sys.path.insert(0, str(ROOT / "scripts" / "challenge2"))
from train_c2_cnn import load_checkpoint, predict_store, write_official_npz  # noqa: E402


def quantiles(values):
    values = np.asarray([v for v in values if v is not None and np.isfinite(v)], float)
    if not len(values):
        return {"count": 0}
    q = np.quantile(values, [0, .05, .25, .5, .75, .95, 1])
    return {"count": int(len(values)), "min": float(q[0]), "p05": float(q[1]), "p25": float(q[2]), "median": float(q[3]),
            "p75": float(q[4]), "p95": float(q[5]), "max": float(q[6]), "mean": float(values.mean())}


def plausibility_frame(task):
    """LCFS + derived scalars on one predicted MAST frame; Thomson chords outside the LCFS."""
    from lcfs import extract_lcfs
    from derive import derive_frame, _points_in_polygon
    grid_R, grid_Z, envelope = pf.load_envelope("MAST")
    psi = np.asarray(task["psi"], np.float64)
    contour = extract_lcfs(psi, grid_R, grid_Z, "MAST", envelope, envelope.astype(float), axis_sign=1.0)
    out = {"shot": task["shot"], "frame": task["frame"], "lcfs_ok": contour is not None}
    derived = derive_frame(psi, grid_R, grid_Z, "MAST", envelope, envelope.astype(float),
                           contour=contour if contour is not None else None, axis_sign=1.0)
    out.update({k: (None if not np.isfinite(v) else float(v)) for k, v in derived.items()})
    pe, ok, cr, cz = (np.asarray(task[k]) for k in ("pe", "pe_mask", "chord_R", "chord_Z"))
    valid = ok & np.isfinite(pe) & np.isfinite(cr) & np.isfinite(cz)
    out["thomson_valid_chords"] = int(valid.sum())
    if contour is not None and valid.any() and pe[valid].max() > 0:
        hot = valid & (pe > 0.1 * pe[valid].max())
        inside = _points_in_polygon(cr[hot], cz[hot], contour)
        out["thomson_hot_chords"] = int(hot.sum())
        out["thomson_hot_outside_fraction"] = float(1.0 - inside.mean()) if hot.any() else None
        cold = valid & (pe <= 0.02 * pe[valid].max())
        if cold.any():
            out["thomson_cold_inside_fraction"] = float(_points_in_polygon(cr[cold], cz[cold], contour).mean())
        out["lcfs_Rmin"], out["lcfs_Rmax"] = float(contour[:, 0].min()), float(contour[:, 0].max())
        out["lcfs_Zmin"], out["lcfs_Zmax"] = float(contour[:, 1].min()), float(contour[:, 1].max())
    return out


def frame_statistics(psi, grid_R, grid_Z, envelope):
    from o_point import find_o_point
    stats = []
    perimeter_idx = np.zeros(psi.shape[1:], bool)
    perimeter_idx[0], perimeter_idx[-1], perimeter_idx[:, 0], perimeter_idx[:, -1] = True, True, True, True
    for frame in psi:
        Ra, Za, iz, ir, used_local = find_o_point(frame.astype(np.float64), grid_R, grid_Z, envelope)
        stats.append((float(frame.mean()), float(frame[iz, ir]), float(frame[perimeter_idx].mean()),
                      float(frame.max() - frame.min()), float(Ra), float(Za), bool(used_local)))
    return stats


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True, help="best.pt or a run directory containing it")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--raw-dir", type=Path, default=ROOT / "data/raw")
    p.add_argument("--max-shots", type=int, default=0, help="first N official shots only (smoke tests)")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--report-frames-per-shot", type=int, default=8)
    p.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    p.add_argument("--npz-name", default="mast_public_test.npz")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.workers)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = args.checkpoint / "best.pt" if args.checkpoint.is_dir() else args.checkpoint
    model, scalar_norm, saved = load_checkpoint(ckpt, device)   # refuses a differing feature spec
    print(f"loaded {ckpt} (epoch {saved.get('epoch')}, feature spec {cd.feature_spec_hash()})", flush=True)
    setup = cd.build_machine_setup("MAST")
    featurizer = cd.FrameFeaturizer(setup, device)
    grid_R, grid_Z, envelope = setup.grid_R, setup.grid_Z, setup.envelope
    paths = cd.mast_parquet_paths(args.raw_dir)
    if args.max_shots:
        paths = paths[:args.max_shots]
    started = time.monotonic()
    arrays, shots, tasks = {}, [], []
    cond_samples = {k: [] for k in cd.SCALAR_COND_NAMES}
    total_frames = 0
    for i, path in enumerate(paths):
        shot = cd.mast_shot_inputs(path, setup)
        store = cd.FrameStore([shot], with_labels=False)
        store_t = store.to_torch(device, False)
        with torch.no_grad():
            entries = featurizer(cd.batch_from_store(store_t, torch.arange(store.n, device=device)))["entries"]
        for k in cond_samples:
            cond_samples[k].append(entries[k].float().cpu().numpy())
        phi_hat, scalars = predict_store(model, featurizer, store_t, store.n, args.batch_size, scalar_norm, device)
        psi = phi_hat.astype(np.float32)  # MAST convention: axis maximum
        arrays[f"shot_{i:04d}_psirz"] = psi
        arrays[f"shot_{i:04d}_q95"] = scalars[:, 0].astype(np.float32)
        arrays[f"shot_{i:04d}_betaN"] = scalars[:, 1].astype(np.float32)
        stats = frame_statistics(psi, grid_R, grid_Z, envelope)
        sel = np.unique(np.linspace(0, store.n - 1, min(args.report_frames_per_shot, store.n)).round().astype(int))
        for t in sel:
            tasks.append({"shot": i, "frame": int(t), "psi": psi[t], "pe": shot.pe[t], "pe_mask": shot.pe_mask[t],
                          "chord_R": shot.chord_R, "chord_Z": shot.chord_Z})
        shots.append({"index": i, "path": path.name, "frames": int(store.n), "Ip_kA_min": float(shot.Ip_kA.min()),
                      "Ip_kA_max": float(shot.Ip_kA.max()), "Ip_sign": shot.ip_sign, "tf_kA_median": float(np.median(shot.tf_kA)),
                      "Ip_unobserved_frames": int((~shot.Ip_observed).sum()), **shot.meta,
                      "pe_valid_fraction": float(shot.pe_mask.mean()),
                      "psi_grid_mean": [s[0] for s in stats], "psi_axis_value": [s[1] for s in stats],
                      "psi_boundary_mean": [s[2] for s in stats], "psi_peak_to_peak": [s[3] for s in stats],
                      "o_point_R": [s[4] for s in stats], "o_point_Z": [s[5] for s in stats],
                      "o_point_local_max_fraction": float(np.mean([s[6] for s in stats])),
                      "q95_pred_median": float(np.median(scalars[:, 0])), "betaN_pred_median": float(np.median(scalars[:, 1]))})
        total_frames += store.n
        if (i + 1) % 100 == 0:
            print(f"predicted {i + 1}/{len(paths)} shots, {total_frames} frames, {time.monotonic() - started:.0f} s", flush=True)
    npz_path = args.output / args.npz_name
    write_official_npz(npz_path, arrays)
    predict_seconds = time.monotonic() - started
    print(f"wrote {npz_path}: {len(paths)} shots, {total_frames} frames in {predict_seconds:.0f} s", flush=True)

    report_started = time.monotonic()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        frames = list(pool.map(plausibility_frame, tasks, chunksize=16))
    per_shot_success = {}
    for f in frames:
        per_shot_success.setdefault(f["shot"], []).append(f["lcfs_ok"])
    for s in shots:
        ok = per_shot_success.get(s["index"], [])
        s["lcfs_success_rate"] = float(np.mean(ok)) if ok else None
    all_stats = {key: quantiles([v for s in shots for v in s[key]]) for key in
                 ("psi_grid_mean", "psi_axis_value", "psi_boundary_mean", "psi_peak_to_peak", "o_point_R", "o_point_Z")}
    derived = {k: quantiles([f.get(k) for f in frames]) for k in ("R_axis", "Z_axis", "kappa", "tri_top", "tri_bot", "volume", "li",
                                                                  "thomson_hot_outside_fraction", "thomson_cold_inside_fraction",
                                                                  "lcfs_Rmin", "lcfs_Rmax", "lcfs_Zmin", "lcfs_Zmax")}
    kappa = np.array([f["kappa"] for f in frames if f.get("kappa") is not None])
    raxis = np.array([f["R_axis"] for f in frames if f.get("R_axis") is not None])
    li = np.array([f["li"] for f in frames if f.get("li") is not None])
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "checkpoint": str(ckpt),
        "checkpoint_epoch": saved.get("epoch"), "checkpoint_val_S": saved.get("validation_score"),
        "checkpoint_synthetic_val_S": saved.get("synthetic_validation_score"), "checkpoint_synthetic_training": saved.get("synthetic"),
        "scalar_norm": {k: v.tolist() for k, v in scalar_norm.items()},
        "feature_spec": cd.feature_spec(), "feature_spec_hash": cd.feature_spec_hash(),
        "conditioning_ranges_mast": {k: quantiles(np.concatenate(v)) for k, v in cond_samples.items()},
        "shots": len(paths), "frames": int(total_frames), "official_order": "sorted pinned manifest paths (package_submission.py)",
        "npz": str(npz_path), "predict_seconds": predict_seconds, "report_seconds": time.monotonic() - report_started,
        "machine_setup": setup.summary(), "device": str(device),
        "report_frames": len(frames), "report_frames_per_shot": args.report_frames_per_shot,
        "lcfs_success_rate_frames": float(np.mean([f["lcfs_ok"] for f in frames])) if frames else None,
        "lcfs_success_rate_per_shot": quantiles([s["lcfs_success_rate"] for s in shots]),
        "shots_with_all_lcfs_ok": int(sum(1 for s in shots if s["lcfs_success_rate"] == 1.0)),
        "shots_with_no_lcfs": int(sum(1 for s in shots if s["lcfs_success_rate"] == 0.0)),
        "derived": derived,
        "plausible_fractions": {
            "R_axis_in_0.8_1.0": float(np.mean((raxis >= 0.8) & (raxis <= 1.0))) if len(raxis) else None,
            "kappa_in_1.8_2.3": float(np.mean((kappa >= 1.8) & (kappa <= 2.3))) if len(kappa) else None,
            "li_in_0.5_1.0": float(np.mean((li >= 0.5) & (li <= 1.0))) if len(li) else None},
        "psi_frame_statistics": all_stats,
        "o_point_local_max_fraction": float(np.mean([s["o_point_local_max_fraction"] for s in shots])),
        "Ip_kA": quantiles([v for s in shots for v in (s["Ip_kA_min"], s["Ip_kA_max"])]),
        "tf_kA_median": quantiles([s["tf_kA_median"] for s in shots]),
        "unobserved_current_frames_total": {k: int(sum(s["unobserved_current_frames"][k] for s in shots)) for k in shots[0]["unobserved_current_frames"]},
        "q95_pred": quantiles([s["q95_pred_median"] for s in shots]), "betaN_pred": quantiles([s["betaN_pred_median"] for s in shots]),
        "label_role": "no MAST labels exist locally; nothing label-derived was read",
    }
    (args.output / "mast_plausibility.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.output / "mast_frames.json").write_text(json.dumps(frames, indent=1) + "\n")
    (args.output / "mast_shots.json").write_text(json.dumps(shots, indent=1) + "\n")
    print(json.dumps({k: report[k] for k in ("shots", "frames", "lcfs_success_rate_frames", "shots_with_all_lcfs_ok", "shots_with_no_lcfs",
                                             "plausible_fractions", "o_point_local_max_fraction")}), flush=True)
    for k in ("R_axis", "Z_axis", "kappa", "volume", "li", "thomson_hot_outside_fraction"):
        print(k, json.dumps(derived[k]), flush=True)
    print("psi stats", json.dumps(all_stats), flush=True)


if __name__ == "__main__":
    main()
