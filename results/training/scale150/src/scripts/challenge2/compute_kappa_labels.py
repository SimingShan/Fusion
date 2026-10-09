#!/usr/bin/env python3
"""Shot-level elongation from DIII-D label LCFS (official scorer extraction) for the kappa stress test.

Train shots: up to --train-frames uniformly spaced cached frames per shot -> per-shot median kappa
-> the lowest --low-fraction of shots by kappa is written as a JSON shot_index list. Dev-fold shots:
up to --val-frames frames per shot -> per-shot median kappa -> two evaluate.py-compatible manifests
(low / high halves of the dev fold by shot-level kappa) with entries pointing back at the native
cache files. Label use: DIII-D train/val labels only.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "starter" / "fusion_scoring"))
from fusion import physics_fields as pf  # noqa: E402


def shot_kappa(task):
    from lcfs import extract_lcfs
    from derive import _shape_from_contour
    grid_R, grid_Z, envelope = pf.load_envelope("DIII-D")
    with np.load(task["path"], allow_pickle=False) as shot:
        n = len(shot["psi"])
        sel = np.unique(np.linspace(0, n - 1, min(task["frames"], n)).round().astype(int))
        psi = np.asarray(shot["psi"][sel], np.float64)
    values = []
    for frame in psi:
        contour = extract_lcfs(-frame, grid_R, grid_Z, "DIII-D", envelope, envelope.astype(float), axis_sign=1.0)
        if contour is None or len(contour) < 8:
            continue
        shape = _shape_from_contour(contour)
        if np.isfinite(shape["kappa"]) and 0.8 <= shape["kappa"] <= 4.0:
            values.append(shape["kappa"])
    return {"shot_index": task["shot_index"], "n_frames_tried": int(len(sel)), "n_valid": len(values),
            "kappa_median": float(np.median(values)) if values else None,
            "kappa_min": float(np.min(values)) if values else None, "kappa_max": float(np.max(values)) if values else None}


def run(entries, data_dir, frames, workers):
    tasks = [{"path": str(data_dir / e["path"]), "shot_index": int(e["shot_index"]), "frames": frames} for e in entries]
    out = []
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, result in enumerate(pool.map(shot_kappa, tasks, chunksize=8)):
            out.append(result)
            if (i + 1) % 500 == 0:
                print(f"{i + 1}/{len(tasks)} shots, {time.monotonic() - started:.0f} s", flush=True)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=Path("data/advanced_v1"))
    p.add_argument("--val-dir", type=Path, default=Path("data/advanced_val_native_v3"))
    p.add_argument("--out", type=Path, default=Path("results/challenge2/c2_cnn_kappa"))
    p.add_argument("--split-dir", type=Path, default=Path("data/c2_cnn_kappa_split"))
    p.add_argument("--train-frames", type=int, default=4)
    p.add_argument("--val-frames", type=int, default=8)
    p.add_argument("--low-fraction", type=float, default=0.6)
    p.add_argument("--max-train-shots", type=int, default=0)
    p.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    train_manifest = json.loads((args.data_dir / "manifest.json").read_text())
    entries = train_manifest["train"]
    if args.max_train_shots:
        entries = entries[:args.max_train_shots]
    train = run(entries, args.data_dir, args.train_frames, args.workers)
    valid = [r for r in train if r["kappa_median"] is not None]
    kappas = np.array([r["kappa_median"] for r in valid])
    threshold = float(np.quantile(kappas, args.low_fraction))
    low = sorted(r["shot_index"] for r in valid if r["kappa_median"] <= threshold)
    (args.out / "train_shot_kappa.json").write_text(json.dumps(train, indent=1) + "\n")
    (args.out / f"train_low_kappa_{int(args.low_fraction * 100)}_shots.json").write_text(json.dumps(low) + "\n")
    summary = {"train_shots": len(train), "train_valid": len(valid), "kappa_threshold": threshold,
               "low_kappa_shots": len(low), "train_kappa_quantiles": dict(zip(["min", "p10", "p25", "p50", "p75", "p90", "max"],
                                                                             np.quantile(kappas, [0, .1, .25, .5, .75, .9, 1]).tolist()))}
    print(json.dumps(summary), flush=True)

    val_manifest = json.loads((args.val_dir / "manifest.json").read_text())
    val_entries = val_manifest["val"]
    val = run(val_entries, args.val_dir, args.val_frames, args.workers)
    (args.out / "val_shot_kappa.json").write_text(json.dumps(val, indent=1) + "\n")
    vk = np.array([r["kappa_median"] if r["kappa_median"] is not None else np.nan for r in val])
    median = float(np.nanmedian(vk))
    halves = {"low": [i for i, k in enumerate(vk) if np.isfinite(k) and k <= median],
              "high": [i for i, k in enumerate(vk) if np.isfinite(k) and k > median]}
    for name, positions in halves.items():
        target = args.split_dir / name
        target.mkdir(parents=True, exist_ok=True)
        rel = os.path.relpath(args.val_dir.resolve(), target.resolve())
        manifest = {"version": val_manifest.get("version"), "feature_names": val_manifest["feature_names"],
                    "scalar_names": val_manifest.get("scalar_names"), "settings_hash": val_manifest.get("settings_hash"),
                    "source": str(args.val_dir), "kappa_split": name, "kappa_median_threshold": median,
                    "source_positions": positions,
                    "val": [{**val_entries[i], "path": os.path.join(rel, val_entries[i]["path"]),
                             "kappa_median": float(vk[i])} for i in positions]}
        (target / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    summary["val"] = {"shots": len(val), "kappa_median_threshold": median, "low_shots": len(halves["low"]), "high_shots": len(halves["high"]),
                      "low_frames": int(sum(val_entries[i]["n_frames"] for i in halves["low"])),
                      "high_frames": int(sum(val_entries[i]["n_frames"] for i in halves["high"])),
                      "val_kappa_quantiles": dict(zip(["min", "p10", "p50", "p90", "max"], np.nanquantile(vk, [0, .1, .5, .9, 1]).tolist()))}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
