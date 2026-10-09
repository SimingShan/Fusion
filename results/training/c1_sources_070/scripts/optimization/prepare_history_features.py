#!/usr/bin/env python3
"""Append input-current history without reopening targets or refitting POD.

Each view entry points at its immutable original shot NPZ and an X_extra NPY
sidecar. Inference must concatenate the original X with this sidecar.
"""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from fusion.data import COILS, arrow_to_numpy, interpolate_signal
from fusion.pod import manifest_hash, read_manifest, standardize_x, validate_splits
from starter.data_fixes import fix_d3d_ip_times
from scripts.optimization.prepare_scaled_features import fit_statistics, _link_or_copy

LAGS_MS = (-200, -500, -1000, -2000, -4000)


def history_feature_names():
    return [f"magnetics_{coil}_{field}_offset_{lag:+d}ms" for lag in LAGS_MS
            for field in ("value", "observed") for coil in COILS]


def extract_history(row, query):
    query = np.asarray(query, dtype=np.float64).ravel()
    shifted = query[:, None] + np.asarray(LAGS_MS)[None, :]
    values = np.empty((len(query), len(LAGS_MS), len(COILS)), dtype=np.float32)
    masks = np.empty_like(values)
    shared = row.get("magnetics_time")
    ip_times = row.get("magnetics_plasma_current_times", shared)
    ip = row.get("magnetics_plasma_current")
    if ip_times is not None and ip is not None and shared is not None:
        if len(ip_times) and len(ip) and np.isfinite(ip).any():
            ip_times = fix_d3d_ip_times({**row, "magnetics_plasma_current_times": ip_times})
    for i, coil in enumerate(COILS):
        times = ip_times if coil == "plasma_current" else row.get(f"magnetics_{coil}_times")
        if times is None:
            times = shared
        v, m = interpolate_signal(times, row.get(f"magnetics_{coil}"), shifted.ravel())
        values[:, :, i], masks[:, :, i] = v.reshape(shifted.shape), m.reshape(shifted.shape)
    result = np.concatenate([part[:, lag, :] for lag in range(len(LAGS_MS)) for part in (values, masks)], axis=1)
    if result.shape != (len(query), 210) or not np.isfinite(result).all():
        raise ValueError("Invalid current-history features")
    return result


def _json(path, value):
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def _task(task):
    import pyarrow.parquet as pq

    extra_path = Path(task["output"]) / task["entry"]["X_extra"]
    if extra_path.exists():
        array = np.load(extra_path, mmap_mode="r", allow_pickle=False)
        if array.shape == (task["entry"]["n_frames"], 210) and array.dtype == np.float32:
            return task["split"], task["entry"]["shot_index"]
    with np.load(task["entry"]["path"], allow_pickle=False) as original:
        # Access only query scaffolding; neither psi nor scalars is loaded.
        query, frames = original["times"], original["frame_indices"]
    raw = pq.ParquetFile(Path(task["raw_dir"]) / task["entry"]["shard"])
    columns = ["efit_times", "magnetics_time"] + [f"magnetics_{coil}" for coil in COILS] + [f"magnetics_{coil}_times" for coil in COILS]
    columns = [name for name in dict.fromkeys(columns) if name in raw.schema_arrow.names]
    batch = next(raw.iter_batches(batch_size=1, columns=columns, use_threads=False))
    row = {name: arrow_to_numpy(batch.column(i)[0]) for i, name in enumerate(columns)}
    if not np.array_equal(np.asarray(row["efit_times"])[frames], query):
        raise ValueError("Current-history shot/frame identity differs from base cache")
    values = extract_history(row, query)
    extra_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = extra_path.with_name(extra_path.name + f".{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.save(handle, values)
    os.replace(temporary, extra_path)
    return task["split"], task["entry"]["shot_index"]


def _consolidate(base_dir, output, manifest, split):
    entries = manifest[split]
    base_path = base_dir / "consolidated" / f"{split}_X.npy"
    original_X = np.load(base_path, mmap_mode="r", allow_pickle=False) if base_path.exists() else None
    n = sum(entry["n_frames"] for entry in entries)
    width = len(manifest["feature_names"])
    if original_X is not None and original_X.shape != (n, width - 210):
        raise ValueError("Base consolidated features disagree with preserved shot/frame selection")
    root = output / "consolidated"
    root.mkdir(exist_ok=True)
    temporary = root / f"{split}_X.partial.npy"
    X = np.lib.format.open_memmap(temporary, mode="w+", dtype="float32", shape=(n, width))
    offset = 0
    for entry in entries:
        end = offset + entry["n_frames"]
        if original_X is not None:
            base_X = original_X[offset:end]
        else:
            with np.load(entry["path"], allow_pickle=False) as archive:
                base_X = archive["X"]
        X[offset:end, :-210] = base_X
        X[offset:end, -210:] = np.load(output / entry["X_extra"], allow_pickle=False)
        offset = end
    X.flush()
    del X
    os.replace(temporary, root / f"{split}_X.npy")
    # Hardlinks do not read or alter the original target arrays.
    for original in (base_dir / "consolidated").glob(f"{split}_*.npy"):
        if original.name != f"{split}_X.npy" and ".partial." not in original.name:
            _link_or_copy(original, root / original.name)
    return {"n_frames": n, "n_shots": len(entries), "feature_count": width,
            "X": f"consolidated/{split}_X.npy", "targets_reused_unchanged": True}


def _prepare_pod(base_dir, output, source_pod, pod_output, manifest, batch_size):
    source_meta = json.loads((source_pod / "basis.json").read_text())
    if source_meta["source_manifest_sha256"] != manifest_hash(base_dir):
        raise ValueError("Source POD differs from the immutable base data")
    if not all(manifest["feature_augmentation"]["complete_splits"].get(split) for split in ("train", "val", "legacy_val")):
        raise ValueError("Prepare complete train/val/legacy feature sidecars before POD normalization")
    with np.load(source_pod / "normalization.npz", allow_pickle=False) as saved:
        norm = {key: saved[key].copy() for key in saved.files}
    base_width = len(manifest["feature_names"]) - 210
    scale = np.concatenate((norm.get("raw_log_scale", np.ones(base_width)), np.ones(210)))
    if len(scale) != len(manifest["feature_names"]):
        raise ValueError("Source log-scale definition differs from base features")
    train = np.load(output / "consolidated/train_X.npy", mmap_mode="r", allow_pickle=False)
    if len(train) != source_meta["training_frames"]:
        raise ValueError("History view changed training frame selection")
    mean, std = fit_statistics(train, scale, batch_size)
    active = np.flatnonzero(std > 1e-6)
    norm.update(raw_log_scale=scale, feature_indices=active,
                x_mean=mean[active].astype(np.float32), x_std=std[active].astype(np.float32))
    pod_output.mkdir(parents=True, exist_ok=True)
    temporary = pod_output / f"normalization.{os.getpid()}.tmp"
    with temporary.open("wb") as handle:
        np.savez(handle, **norm)
    os.replace(temporary, pod_output / "normalization.npz")
    for split in ("train", "val", "legacy_val"):
        X = train if split == "train" else np.load(output / "consolidated" / f"{split}_X.npy", mmap_mode="r", allow_pickle=False)
        temporary = pod_output / f"{split}_X.partial.npy"
        projected = np.lib.format.open_memmap(temporary, mode="w+", dtype="float32", shape=(len(X), len(active)))
        for first in range(0, len(X), batch_size):
            projected[first:first + batch_size] = standardize_x(X[first:first + batch_size], norm)
        projected.flush()
        del projected
        os.replace(temporary, pod_output / f"{split}_X.npy")
        for key in ("coefficients", "scalars"):
            _link_or_copy(source_pod / f"{split}_{key}.npy", pod_output / f"{split}_{key}.npy")
        print(f"History POD features normalized: {split} {len(X)} frames", flush=True)
    if (source_pod / "basis_sample_indices.npy").exists():
        _link_or_copy(source_pod / "basis_sample_indices.npy", pod_output / "basis_sample_indices.npy")
    metadata = {**source_meta, "source_manifest_sha256": manifest_hash(output), "data_dir": str(output),
                "input_features": len(manifest["feature_names"]), "active_features": len(active),
                "feature_transform": {"name": "physical_scaled_Thomson_plus_current_history", "fit_split": "train", "complete": True},
                "history_view": {"lags_ms": list(LAGS_MS), "added_raw_features": 210,
                                 "source_pod": str(source_pod), "source_basis_metadata": source_meta,
                                 "source_normalization_sha256": hashlib.sha256((source_pod / "normalization.npz").read_bytes()).hexdigest(),
                                 "basis_refitted": False, "target_arrays_reused_unchanged": True,
                                 "audit_features_prepared": False, "audit_labels_read": False}}
    _json(pod_output / "basis.json", metadata)
    return metadata


def prepare(args):
    started = time.monotonic()
    base_dir, output, raw_dir = args.base_dir.resolve(), args.output.resolve(), args.raw_dir.resolve()
    if base_dir == output:
        raise ValueError("History features require a separate view directory")
    base = read_manifest(base_dir)
    if "feature_augmentation" in base:
        raise ValueError("Supply the unaugmented base cache")
    configuration = {"base_dir": str(base_dir), "base_manifest_sha256": manifest_hash(base_dir),
                     "lags_ms": list(LAGS_MS), "source_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        manifest = read_manifest(output)
        if manifest["feature_augmentation"]["configuration"] != configuration:
            raise ValueError("Existing history cache has a different source/configuration")
    else:
        manifest = json.loads(json.dumps(base))
        manifest["feature_names"] += history_feature_names()
        manifest["feature_augmentation"] = {"configuration": configuration, "base_feature_count": len(base["feature_names"]),
                                             "extra_feature_count": 210, "complete_splits": {}, "target_values_read": False}
        manifest["consolidated"] = {}
        manifest["complete_splits"] = {split: False for split in ("train", "val", "legacy_val", "audit", "test")}
        for split in ("train", "val", "legacy_val", "audit", "test"):
            for entry in manifest.get(split, []):
                entry["path"] = str((base_dir / entry["path"]).resolve())
                if split != "audit":
                    entry["X_extra"] = f"history/{split}/shot_{entry['shot_index']:05d}.npy"
        _json(manifest_path, manifest)
    tasks = [{"output": str(output), "raw_dir": str(raw_dir), "split": split, "entry": entry}
             for split in args.splits for entry in manifest[split]
             if not (output / entry["X_extra"]).exists()]
    if args.limit_shots:
        tasks = tasks[:args.limit_shots]
    print(f"History feature tasks: {len(tasks)}, workers={args.workers}, added_features=210", flush=True)
    completed = 0
    if args.workers == 1:
        for task in tasks:
            _task(task)
            completed += 1
    else:
        iterator = iter(tasks)
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            pending = {}
            for _ in range(min(len(tasks), args.workers * 2)):
                task = next(iterator)
                pending[executor.submit(_task, task)] = task
            while pending:
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    pending.pop(future)
                    future.result()
                    completed += 1
                    if completed % 128 == 0:
                        print(f"Prepared history: {completed}/{len(tasks)} new shots ({time.monotonic()-started:.1f}s)", flush=True)
                    task = next(iterator, None)
                    if task is not None:
                        pending[executor.submit(_task, task)] = task
    for split in args.splits:
        complete = bool(manifest[split]) and all((output / entry["X_extra"]).exists() for entry in manifest[split])
        manifest["feature_augmentation"]["complete_splits"][split] = complete
        manifest["complete_splits"][split] = complete
        if complete:
            manifest["consolidated"][split] = _consolidate(base_dir, output, manifest, split)
    _json(manifest_path, manifest)
    if not args.no_pod:
        _prepare_pod(base_dir, output, args.source_pod.resolve(), args.pod_output.resolve(), manifest, args.batch_size)
    print(json.dumps({"output": str(output), "complete_splits": manifest["feature_augmentation"]["complete_splits"],
                      "elapsed_seconds": time.monotonic() - started}), flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--base-dir", type=Path, default=Path("data/advanced_v1"))
    parser.add_argument("--output", type=Path, default=Path("data/advanced_history_v2"))
    parser.add_argument("--source-pod", type=Path, default=Path("data/pod_scaled_v2"))
    parser.add_argument("--pod-output", type=Path, default=Path("data/pod_history_v2"))
    parser.add_argument("--splits", nargs="+", choices=("train", "val", "legacy_val", "test"), default=["legacy_val", "val", "train"])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--limit-shots", type=int, default=0)
    parser.add_argument("--no-pod", action="store_true")
    args = parser.parse_args()
    if args.workers < 1 or args.batch_size < 1 or args.limit_shots < 0:
        parser.error("Workers/batch size must be positive and shot limit nonnegative")
    prepare(args)


if __name__ == "__main__":
    main()
