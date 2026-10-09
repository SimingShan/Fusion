#!/usr/bin/env python3
"""Tune a small mean/shape prediction blend on an existing validation fold.

Only model predictions enter a blended field. Validation labels diagnose errors
and choose a global blend weight; they never supply per-frame offsets or shape.
The tuning fold is not an independent estimate of the selected blend's quality.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from fusion.pod import split_entries


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def blend_fields(a, b, mean_weight, shape_weight):
    """Weights are fractions of source A; independent of labels at inference."""
    am = a.mean(axis=(-2, -1), keepdims=True, dtype=np.float64)
    bm = b.mean(axis=(-2, -1), keepdims=True, dtype=np.float64)
    return (shape_weight * (a - am) + (1.0 - shape_weight) * (b - bm)
            + mean_weight * am + (1.0 - mean_weight) * bm).astype(np.float32)


def scalar_r2(truth, prediction):
    result = []
    for column in range(2):
        valid = np.isfinite(truth[:, column])
        y = truth[valid, column].astype(np.float64)
        p = prediction[valid, column].astype(np.float64)
        denominator = np.square(y - y.mean()).sum()
        result.append(float(1 - np.square(y - p).sum() / denominator))
    return dict(q95=result[0], betaN=result[1], mean=float(np.mean(result)))


def load_predictions(path, entries):
    expected = {f"shot_{i:04d}_{name}" for i in range(len(entries))
                for name in ("psirz", "q95", "betaN")}
    values = []
    before = sha256(path)
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != expected:
            raise ValueError(f"Prediction inventory mismatch: {path}")
        for i, entry in enumerate(entries):
            field = archive[f"shot_{i:04d}_psirz"]
            scalars = np.column_stack([archive[f"shot_{i:04d}_{name}"] for name in ("q95", "betaN")])
            if field.shape != (entry["n_frames"], 65, 65) or scalars.shape != (entry["n_frames"], 2):
                raise ValueError(f"Prediction shape mismatch in {path}, shot {i}")
            if not np.isfinite(field).all() or not np.isfinite(scalars).all():
                raise ValueError(f"Nonfinite source prediction: {path}, shot {i}")
            values.append((field, scalars))
    if before != sha256(path):
        raise ValueError(f"Source predictions changed while reading: {path}")
    return values, before


def diagnose(entries, truths, sources, output):
    pooled_truth = np.concatenate([v[0].ravel() for v in truths]).astype(np.float64)
    denominator = np.square(pooled_truth - pooled_truth.mean()).sum()
    total_cells = len(pooled_truth)
    del pooled_truth
    reports, pixel_maps = {}, {}
    for name, values in sources.items():
        per_shot, total_sse, mean_sse, centered_sse = [], 0.0, 0.0, 0.0
        pixel_sse, frames = np.zeros((65, 65), dtype=np.float64), 0
        for entry, (truth, _), (pred, _) in zip(entries, truths, values):
            error = pred.astype(np.float64) - truth
            mean_error = error.mean(axis=(1, 2), keepdims=True)
            local_sse = np.square(error).sum()
            local_mean = np.square(mean_error).sum() * 4225
            local_centered = np.square(error - mean_error).sum()
            np.testing.assert_allclose(local_sse, local_mean + local_centered, rtol=1e-11)
            total_sse += local_sse
            mean_sse += local_mean
            centered_sse += local_centered
            pixel_sse += np.square(error).sum(axis=0)
            frames += len(error)
            per_shot.append({"shot_index": entry["shot_index"], "shot_id": entry.get("shot_id"),
                             "frames": len(error), "sse": float(local_sse),
                             "spatial_mean_sse": float(local_mean), "centered_shape_sse": float(local_centered),
                             "spatial_mean_fraction": float(local_mean / local_sse),
                             "mean_offset_rmse": float(np.sqrt(np.square(mean_error).mean())),
                             "centered_shape_rmse": float(np.sqrt(local_centered / error.size))})
        reports[name] = {"flux_r2": float(1 - total_sse / denominator),
                         "flux_rmse": float(np.sqrt(total_sse / total_cells)),
                         "spatial_mean_sse": float(mean_sse), "centered_shape_sse": float(centered_sse),
                         "spatial_mean_fraction": float(mean_sse / total_sse),
                         "mean_offset_rmse": float(np.sqrt(mean_sse / total_cells)),
                         "centered_shape_rmse": float(np.sqrt(centered_sse / total_cells)),
                         "per_shot": per_shot}
        pixel_maps[f"{name}_pixel_mse"] = pixel_sse / frames
    np.savez_compressed(output / "pixel_error_maps.npz", **pixel_maps)
    return reports


def write_blend(path, a, b, mean_weight, shape_weight, scalar_weight):
    temporary = path.with_suffix(".tmp.npz")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=1, allowZip64=True) as archive:
        for i, ((af, av), (bf, bv)) in enumerate(zip(a, b)):
            field = blend_fields(af, bf, mean_weight, shape_weight)
            scalars = (scalar_weight * av + (1.0 - scalar_weight) * bv).astype(np.float32)
            for key, array in (("psirz", field), ("q95", scalars[:, 0]), ("betaN", scalars[:, 1])):
                with archive.open(f"shot_{i:04d}_{key}.npy", "w", force_zip64=True) as stream:
                    np.lib.format.write_array(stream, array, allow_pickle=False)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-a", type=Path, required=True)
    parser.add_argument("--source-b", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--split", choices=["val", "legacy_val"], default="val")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--weights", type=float, nargs="+", default=[0.0, 0.5, 1.0])
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if any(not 0 <= w <= 1 for w in args.weights):
        parser.error("Blend weights must lie in [0, 1]")
    args.output.mkdir(parents=True, exist_ok=True)
    entries = split_entries(args.data_dir, args.split)
    truths = []
    for entry in entries:
        with np.load(args.data_dir / entry["path"], allow_pickle=False) as shot:
            truths.append((shot["psi"], shot["scalars"]))
    a, hash_a = load_predictions(args.source_a, entries)
    b, hash_b = load_predictions(args.source_b, entries)
    diagnostics = diagnose(entries, truths, {"source_a": a, "source_b": b}, args.output)
    scalar_truth = np.concatenate([v[1] for v in truths])
    scalar_a = np.concatenate([v[1] for v in a])
    scalar_b = np.concatenate([v[1] for v in b])
    scalar_candidates = [{"source_a_weight": weight,
                          **scalar_r2(scalar_truth, weight * scalar_a + (1.0 - weight) * scalar_b)}
                         for weight in args.weights]
    best_scalar = max(scalar_candidates, key=lambda value: max(0.0, value["mean"]))
    report = {"created_utc": datetime.now(timezone.utc).isoformat(),
              "evaluation_type": "validation_tuned_prediction_blend",
              "independent_holdout": False, "audit_labels_read": False, "leaderboard_submission": False,
              "weight_definition": "All weights are fractions of source A; source B receives 1 minus weight.",
              "formula": "flux = shape_weight*(A-mean(A)) + (1-shape_weight)*(B-mean(B)) + mean_weight*mean(A) + (1-mean_weight)*mean(B); means are spatial per frame",
              "label_use": "Validation labels diagnose errors and select global candidate weights only. No per-frame ground-truth offsets or fields enter predictions.",
              "data_dir": str(args.data_dir.resolve()), "split": args.split,
              "source_a": {"path": str(args.source_a.resolve()), "sha256": hash_a},
              "source_b": {"path": str(args.source_b.resolve()), "sha256": hash_b},
              "shots": len(entries), "frames": sum(v["n_frames"] for v in entries),
              "weights": args.weights, "diagnostics": diagnostics,
              "scalar_candidates": scalar_candidates, "selected_scalar": best_scalar,
              "scalar_selection": "Exact pooled scalar R2; the scalar term is independent of flux/shape and selected before the nine flux scores.",
              "candidates": []}
    summary_path = args.output / "summary.json"
    summary_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"source_diagnostics": {name: {k: v for k, v in d.items() if k != "per_shot"}
                                              for name, d in diagnostics.items()},
                      "scalar_candidates": scalar_candidates, "selected_scalar": best_scalar}), flush=True)
    for mean_weight in args.weights:
        for shape_weight in args.weights:
            name = f"mean_a{mean_weight:g}_shape_a{shape_weight:g}".replace(".", "p")
            directory = args.output / name
            directory.mkdir(exist_ok=True)
            prediction_path, score_path = directory / "val_predictions.npz", directory / "score.json"
            write_blend(prediction_path, a, b, mean_weight, shape_weight, best_scalar["source_a_weight"])
            with (directory / "score.log").open("w") as log:
                subprocess.run([sys.executable, str(ROOT / "scripts/evaluate.py"),
                                "--data-dir", str(args.data_dir), "--split", args.split,
                                "--pred", str(prediction_path), "--output", str(score_path),
                                "--workers", str(args.workers)], stdout=log, stderr=subprocess.STDOUT, check=True)
            score = json.loads(score_path.read_text())
            candidate = {"name": name, "mean_weight_a": mean_weight, "shape_weight_a": shape_weight,
                         "scalar_weight_a": best_scalar["source_a_weight"],
                         **{key: score[key] for key in ("S", "r2_psi", "r2_qb", "dlcfs", "consistency", "r2_each", "r2_cons_each")},
                         "prediction_path": str(prediction_path.resolve()), "score_path": str(score_path.resolve())}
            report["candidates"].append(candidate)
            report["best"] = max(report["candidates"], key=lambda value: value["S"])
            summary_path.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(candidate), flush=True)
    print("WINNER " + json.dumps(report["best"]), flush=True)
    best = report["best"]
    lines = ["Validation blend experiment", "", "This is model selection on the original 32-shot validation fold; the selected score is not an independent holdout or leaderboard score.", "",
             f"Source A: {args.source_a.resolve()}", f"Source B: {args.source_b.resolve()}", "",
             "Per-frame spatial means are computed from predictions only. Validation labels select global weights; no true field value enters a blend.", "",
             f"Best S={best['S']:.6f}; A weights: mean={best['mean_weight_a']:g}, centered shape={best['shape_weight_a']:g}, scalars={best['scalar_weight_a']:g}.", "",
             "See summary.json for all candidates, source hashes, scalar R2, and per-shot mean/shape SSE decomposition. pixel_error_maps.npz contains per-pixel mean squared errors for both source models."]
    (args.output / "README.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
