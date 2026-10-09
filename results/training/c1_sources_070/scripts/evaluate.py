#!/usr/bin/env python3
"""Score cached held-out DIII-D predictions with the unmodified official scorer.

Example: python scripts/evaluate.py --data-dir data/processed \
    --pred results/deeponet/validation_predictions.npz \
    --output results/deeponet/score.json --workers 8

Zero sampling limits mean all cached validation data. A cache may itself contain
sampled frames; the report distinguishes cached, evaluated, and original counts.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "starter"))
import local_score as official  # noqa: E402


def scorer_fingerprint():
    paths = [ROOT / "starter/local_score.py"]
    paths += sorted((ROOT / "starter/fusion_scoring").glob("*.py"))
    paths += [ROOT / "starter/fusion_scoring/masks/d3d_envelope.npz"]
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def uniform_indices(n, cap):
    return np.arange(n) if not cap or cap >= n else np.linspace(0, n - 1, cap, dtype=int)


def load_task(task):
    """Read the selected, unaltered target frames and their matching predictions."""
    with np.load(task["path"], allow_pickle=False) as shot:
        n = len(shot["psi"])
        indices = uniform_indices(n, task["max_frames"])
        gt = {"psi": np.asarray(shot["psi"][indices], dtype=np.float32),
              "q95": np.asarray(shot["scalars"][indices, 0], dtype=np.float64),
              "betaN": np.asarray(shot["scalars"][indices, 1], dtype=np.float64)}
        grid_R = np.asarray(shot["R"], dtype=np.float64)
        grid_Z = np.asarray(shot["Z"], dtype=np.float64)
        original_frames = (shot["frame_indices"][indices] if "frame_indices" in shot
                           else indices).astype(int).tolist()
    if gt["psi"].shape != (len(indices), 65, 65):
        raise ValueError(f"{task['path']}: invalid flux shape {gt['psi'].shape}")
    if not np.isfinite(gt["psi"]).all():
        raise ValueError(f"{task['path']}: official local scorer requires finite DIII-D targets")
    with np.load(ROOT / "starter/fusion_scoring/masks/d3d_envelope.npz") as mask:
        R, Z = mask["grid_R"], mask["grid_Z"]
        coarse = mask["mask_coarse"].astype(bool)
    if not (np.allclose(grid_R, R, rtol=1e-6, atol=1e-7)
            and np.allclose(grid_Z, Z, rtol=1e-6, atol=1e-7)):
        raise ValueError(f"{task['path']}: physical grid differs from the official DIII-D mask")
    if task["perfect"]:
        pred = {"psirz": gt["psi"].copy(), "q95": gt["q95"].copy(),
                "betaN": gt["betaN"].copy()}
    else:
        pred = {}
        with np.load(task["pred"], allow_pickle=False) as archive:
            for name in ("psirz", "q95", "betaN"):
                key = f"shot_{task['validation_index']:04d}_{name}"
                if key not in archive:
                    raise ValueError(f"Missing prediction key: {key}")
                value = archive[key]
                shape = (n, 65, 65) if name == "psirz" else (n,)
                if value.shape != shape:
                    raise ValueError(f"{key}: expected {shape}, got {value.shape}")
                if not np.issubdtype(value.dtype, np.floating):
                    raise ValueError(f"{key}: floating dtype required")
                pred[name] = value[indices]
    return gt, pred, R, Z, coarse, coarse.astype(np.float64), original_frames


def reference_path(task, gt, R, Z, original_frames):
    digest = hashlib.sha256(task["fingerprint"].encode())
    for value in (gt["psi"], R, Z, np.asarray(original_frames, dtype=np.int64)):
        digest.update(np.ascontiguousarray(value).tobytes())
    return Path(task["cache_dir"]) / f"reference_{digest.hexdigest()}.npz"


def save_reference(path, ref):
    contours, cons, valid, rgeo = ref
    lengths = np.asarray([0 if c is None else len(c) for c in contours], dtype=np.int64)
    points = [c for c in contours if c is not None]
    flat = np.concatenate(points, axis=0) if points else np.empty((0, 2), dtype=np.float64)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temp.open("wb") as handle:
        np.savez_compressed(handle, contour_points=flat, contour_lengths=lengths,
                            consistency=cons, valid=valid, rgeo=np.asarray(rgeo))
    os.replace(temp, path)


def read_reference(path):
    with np.load(path, allow_pickle=False) as data:
        points, lengths = data["contour_points"], data["contour_lengths"]
        offsets = np.concatenate(([0], np.cumsum(lengths)))
        contours = [points[offsets[i]:offsets[i + 1]] if length else None
                    for i, length in enumerate(lengths)]
        return contours, data["consistency"], data["valid"], float(data["rgeo"])


def moment(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(values.sum()), float(np.square(values).sum()), int(values.size)


def reference_worker(task):
    gt, pred, R, Z, coarse, fine, frames = load_task(task)
    cache = reference_path(task, gt, R, Z, frames)
    hit = cache.exists()
    if hit:
        ref = read_reference(cache)
    else:
        ref = official.build_reference(gt["psi"], R, Z, coarse, fine)
        save_reference(cache, ref)
    sums = {}
    sums["psi_sum"], sums["psi_sumsq"], sums["psi_n"] = moment(gt["psi"])
    for prefix, values in (("scal", [gt[name] for name in official.SCALARS]),
                           ("cons", [ref[1][:, j] for j in range(official.N_CONS)])):
        columns = np.asarray([moment(value) for value in values], dtype=np.float64)
        for column, suffix in enumerate(("sum", "sumsq", "n")):
            sums[f"{prefix}_{suffix}"] = columns[:, column]
    # Predictions have been shape-checked. Compute the same fold-global sign
    # decision as local_score.py, retaining sums before choosing the sign.
    residuals, _, _ = official.psi_residuals(gt["psi"], pred["psirz"], 0.0)
    # Nonfinite predictions are replaced by the global truth mean only after it
    # is known. Store sufficient statistics for that correction to both signs.
    bad = ~np.isfinite(pred["psirz"])
    return {"validation_index": task["validation_index"], "cache": str(cache),
            "cache_hit": hit, "stats": sums, "sign_residuals": residuals,
            "nonfinite_n": int(bad.sum()), "nonfinite_gt_sum": float(gt["psi"][bad].sum()),
            "frames": frames}


def score_worker(task, prepared, psi_sign, means):
    gt, pred, R, Z, coarse, fine, _ = load_task(task)
    return official.score_shot(gt, read_reference(prepared["cache"]), pred,
                               R, Z, coarse, fine, psi_sign, means)


def parallel_map(function, jobs, workers, stage):
    results = [None] * len(jobs)
    if workers == 1:
        for index, args in enumerate(jobs):
            results[index] = function(*args)
            print(f"{stage}: {index + 1}/{len(jobs)} shots", flush=True)
        return results
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(function, *args): index for index, args in enumerate(jobs)}
        for completed, future in enumerate(as_completed(futures), 1):
            results[futures[future]] = future.result()
            print(f"{stage}: {completed}/{len(jobs)} shots", flush=True)
    return results


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--split", choices=["val", "legacy_val", "audit"], default="val")
    parser.add_argument("--pred", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-shots", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=0,
                        help="Uniformly select this many existing frames per shot; 0 = all cached")
    parser.add_argument("--perfect", action="store_true",
                        help="Self-check using ground truth as predictions, never a model result")
    args = parser.parse_args()
    if args.workers < 1 or min(args.max_shots, args.max_frames) < 0:
        parser.error("workers must be positive and sampling limits nonnegative")
    if not args.perfect and args.pred is None:
        parser.error("--pred is required unless --perfect")
    started = time.monotonic()
    manifest = json.loads((args.data_dir / "manifest.json").read_text())
    entries = manifest.get("splits", manifest)[args.split]
    if not entries:
        raise ValueError("Validation split is empty")
    selected = uniform_indices(len(entries), args.max_shots)
    cache_dir = args.data_dir / "scoring_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = scorer_fingerprint()
    tasks = [{"path": str((args.data_dir / entries[i]["path"]).resolve()),
              "validation_index": int(i), "max_frames": args.max_frames,
              "cache_dir": str(cache_dir.resolve()), "fingerprint": fingerprint,
              "pred": str(args.pred.resolve()) if args.pred else None, "perfect": args.perfect}
             for i in selected]
    print(f"Local held-out DIII-D evaluation; official metric {official.SCORING_VERSION}; "
          f"{len(tasks)}/{len(entries)} cached validation shots", flush=True)
    prepared = parallel_map(reference_worker, [(task,) for task in tasks], args.workers, "Reference")
    stats = {key: sum(item["stats"][key] for item in prepared) for key in prepared[0]["stats"]}
    means = (stats["psi_sum"] / stats["psi_n"],
             np.divide(stats["scal_sum"], stats["scal_n"], out=np.zeros(official.N_SCALARS),
                       where=stats["scal_n"] > 0),
             np.divide(stats["cons_sum"], stats["cons_n"], out=np.zeros(official.N_CONS),
                       where=stats["cons_n"] > 0))
    totals = {sign: sum(item["sign_residuals"][sign]
                        + item["nonfinite_n"] * means[0] ** 2
                        - 2 * means[0] * item["nonfinite_gt_sum"] for item in prepared)
              for sign in official.PSI_SIGNS}
    psi_sign = min(official.PSI_SIGNS, key=lambda sign: totals[sign])
    parts = parallel_map(score_worker,
                         [(task, prep, psi_sign, means) for task, prep in zip(tasks, prepared)],
                         args.workers, "Score")
    accumulator = official.Accum()
    accumulator.psi_sign = psi_sign
    for part in parts:
        accumulator.add(part)
    result = official.finalize_machine(accumulator, stats)
    evaluated_frames = sum(len(item["frames"]) for item in prepared)
    cached_frames = sum(int(entry["n_frames"]) for entry in entries)
    original_frames = sum(int(entry.get("raw_frames", entry["n_frames"])) for entry in entries)
    sampled = len(tasks) < len(entries) or evaluated_frames < original_frames
    result["evaluation"] = {
        "type": "perfect_prediction_self_check" if args.perfect else "local_held_out_validation",
        "leaderboard_submission": False, "machine": "DIII-D", "split": args.split,
        "scorer_version": official.SCORING_VERSION, "scorer_sha256": fingerprint,
        "sampled": sampled, "selection": "deterministic uniform existing frames and shots; no interpolation",
        "available_validation_shots": len(entries), "evaluated_shots": len(tasks),
        "cached_validation_frames": cached_frames, "original_validation_frames": original_frames,
        "evaluated_frames": evaluated_frames, "max_shots": args.max_shots,
        "max_frames_per_shot": args.max_frames, "reference_cache_hits": sum(p["cache_hit"] for p in prepared),
        "prediction_path": str(args.pred.resolve()) if args.pred else None,
        "data_directory": str(args.data_dir.resolve()), "workers": args.workers,
        "elapsed_seconds": time.monotonic() - started,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "shots": [{"validation_index": int(index), "shot_index": entries[index].get("shot_index"),
                   "shot_id": entries[index].get("shot_id"), "cached_frames": entries[index]["n_frames"],
                   "original_frames": entries[index].get("raw_frames", entries[index]["n_frames"]),
                   "evaluated_original_frame_indices": item["frames"]}
                  for index, item in zip(selected, prepared)]}
    if args.perfect:
        result["self_check_passed"] = abs(result["S"] - 1.0) < 1e-4
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(json_safe(result), indent=2, allow_nan=False) + "\n")
    print(f"S={result['S']:.6f}; R²ψ={result['r2_psi']:.6f}; "
          f"{evaluated_frames} evaluated frames; wrote {args.output}", flush=True)
    if args.perfect and not result["self_check_passed"]:
        raise SystemExit("Perfect-prediction self-check did not reach S=1; inspect degenerate scalars/sample size")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
