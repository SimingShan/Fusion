#!/usr/bin/env python3
"""Reuse a train-only POD basis with physically scaled Thomson log features.

Run after the source POD preparation job completes. Target arrays are reused
without modification; only feature normalization is fitted and projected.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from fusion.pod import load_arrays, manifest_hash, read_manifest, split_entries, validate_splits


def log_scales(names: list[str], ne_scale: float = 1e19, te_scale: float = 1e3) -> np.ndarray:
    scale = np.ones(len(names), dtype=np.float64)
    for index, name in enumerate(names):
        if "log1p" in name and "_ne_" in name:
            scale[index] = ne_scale
        elif "log1p" in name and "_Te_" in name:
            scale[index] = te_scale
    return scale


def transform_raw(X: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Transform stored log1p(v) to log1p(v / physical_scale), in float64."""
    values = np.asarray(X, dtype=np.float64).copy()
    changed = np.flatnonzero(scale != 1)
    if changed.size:
        values[:, changed] = np.log1p(np.expm1(values[:, changed]) / scale[changed])
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite scaled features")
    return values


def fit_statistics(X: np.ndarray, scale: np.ndarray, batch_size: int = 8192) -> tuple[np.ndarray, np.ndarray]:
    """Merge batch central moments without cancellation on constant geometry."""
    count = 0
    mean = np.zeros(X.shape[1], dtype=np.float64)
    m2 = np.zeros_like(mean)
    for first in range(0, len(X), batch_size):
        values = transform_raw(X[first:first + batch_size], scale)
        n = len(values)
        batch_mean = values.mean(axis=0)
        batch_m2 = np.square(values - batch_mean).sum(axis=0)
        delta = batch_mean - mean
        total = count + n
        mean += delta * n / total
        m2 += batch_m2 + np.square(delta) * count * n / total
        count = total
    if not count:
        raise ValueError("Empty training features")
    return mean, np.sqrt(np.maximum(m2 / count, 0))


def standardize_scaled(X: np.ndarray, norm: dict[str, np.ndarray]) -> np.ndarray:
    indices = norm["feature_indices"]
    values = np.asarray(X[:, indices], dtype=np.float64).copy()
    scales = norm["raw_log_scale"][indices]
    changed = scales != 1
    values[:, changed] = np.log1p(np.expm1(values[:, changed]) / scales[changed])
    return np.clip((values - norm["x_mean"]) / norm["x_std"], -12, 12).astype(np.float32)


def _json_atomic(path: Path, data: dict) -> None:
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    os.replace(temporary, path)


def _link_or_copy(source: Path, destination: Path) -> str:
    if destination.exists() and os.path.samefile(source, destination):
        return "hardlink"
    temporary = destination.with_name(destination.name + f".{os.getpid()}.tmp")
    if temporary.exists():
        temporary.unlink()
    try:
        os.link(source, temporary)
        method = "hardlink"
    except OSError:
        shutil.copy2(source, temporary)
        method = "copy"
    os.replace(temporary, destination)
    return method


def prepare(args: argparse.Namespace) -> dict:
    started = time.monotonic()
    data_dir, source, output = args.data_dir.resolve(), args.source_pod.resolve(), args.output.resolve()
    if output == source:
        raise ValueError("Scaled features must use a separate output directory")
    validate_splits(data_dir)
    manifest = read_manifest(data_dir)
    names = manifest["feature_names"]
    source_meta_path = source / "basis.json"
    source_norm_path = source / "normalization.npz"
    source_meta = json.loads(source_meta_path.read_text())
    if source_meta["source_manifest_sha256"] != manifest_hash(data_dir):
        raise ValueError("Source POD provenance disagrees with the selected training cache")
    with np.load(source_norm_path, allow_pickle=False) as saved:
        norm = {key: saved[key].copy() for key in saved.files}
    if "raw_log_scale" in norm:
        raise ValueError("Source POD already uses transformed features; provide the original source POD")
    scale = log_scales(names, args.ne_scale, args.te_scale)
    fingerprint_data = {"source_basis_sha256": hashlib.sha256(source_meta_path.read_bytes()).hexdigest(),
                        "source_normalization_sha256": hashlib.sha256(source_norm_path.read_bytes()).hexdigest(),
                        "feature_names": names, "ne_scale": args.ne_scale, "te_scale": args.te_scale,
                        "variance_threshold": args.variance_threshold,
                        "source_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    fingerprint = hashlib.sha256(json.dumps(fingerprint_data, sort_keys=True).encode()).hexdigest()
    if (output / "basis.json").exists():
        previous = json.loads((output / "basis.json").read_text())
        if previous.get("feature_transform", {}).get("fingerprint") != fingerprint:
            raise ValueError("Existing scaled output has different settings/source; use a new output directory")
        if previous.get("feature_transform", {}).get("complete"):
            return previous
    output.mkdir(parents=True, exist_ok=True)
    train = load_arrays(data_dir, "train", keys=("X",))["X"]
    if train.shape[1] != len(names) or len(train) != source_meta["training_frames"]:
        raise ValueError("Source POD and training feature dimensions disagree")
    mean, std = fit_statistics(train, scale, args.batch_size)
    limit = int(source_meta.get("feature_limit", 0)) or len(names)
    active = np.flatnonzero((std > args.variance_threshold) & (np.arange(len(names)) < limit))
    norm.update(raw_log_scale=scale, feature_indices=active,
                x_mean=mean[active].astype(np.float32), x_std=std[active].astype(np.float32))
    if not all(np.isfinite(value).all() for value in norm.values()):
        raise ValueError("Nonfinite normalization")
    norm_temporary = output / f"normalization.{os.getpid()}.tmp"
    with norm_temporary.open("wb") as handle:
        np.savez(handle, **norm)
    os.replace(norm_temporary, output / "normalization.npz")
    projections = {}
    for split in ("train", "val", "legacy_val"):
        entries = split_entries(data_dir, split)
        if not entries:
            continue
        X = train if split == "train" else load_arrays(data_dir, split, keys=("X",))["X"]
        n = sum(entry["n_frames"] for entry in entries)
        targets = {}
        for key, width in (("coefficients", len(norm["basis"])), ("scalars", 2)):
            path = source / f"{split}_{key}.npy"
            target = np.load(path, mmap_mode="r", allow_pickle=False)
            if target.shape != (n, width) or target.dtype != np.float32:
                raise ValueError(f"Unexpected immutable source target shape/dtype: {path}")
            targets[key] = _link_or_copy(path, output / path.name)
        temporary = output / f"{split}_X.partial.npy"
        projected = np.lib.format.open_memmap(temporary, mode="w+", dtype="float32", shape=(n, len(active)))
        for first in range(0, n, args.batch_size):
            projected[first:first + args.batch_size] = standardize_scaled(X[first:first + args.batch_size], norm)
        projected.flush()
        del projected
        os.replace(temporary, output / f"{split}_X.npy")
        projections[split] = {"n_frames": n, "active_features": len(active), "target_reuse": targets}
        print(f"Scaled {split}: {n} frames, {len(active)} active features", flush=True)
    if (source / "basis_sample_indices.npy").exists():
        _link_or_copy(source / "basis_sample_indices.npy", output / "basis_sample_indices.npy")
    metadata = {**source_meta, "active_features": len(active), "input_features": len(names),
                "feature_transform": {"name": "physical_scale_before_log1p", "formula": "log1p(expm1(stored_log1p) / physical_scale)",
                                      "ne_scale": args.ne_scale, "Te_scale": args.te_scale,
                                      "transformed_feature_count": int(np.count_nonzero(scale != 1)),
                                      "fit_split": "train", "target_arrays_reused_unchanged": True,
                                      "source_pod": str(source), "fingerprint": fingerprint,
                                      **fingerprint_data, "projections": projections,
                                      "seconds": time.monotonic() - started, "complete": True}}
    _json_atomic(output / "basis.json", metadata)
    print(json.dumps({"output": str(output), "active_features": len(active),
                      "transformed_features": int(np.count_nonzero(scale != 1)),
                      "seconds": time.monotonic() - started}), flush=True)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/advanced_v1"))
    parser.add_argument("--source-pod", type=Path, default=Path("data/pod_full_v1"))
    parser.add_argument("--output", type=Path, default=Path("data/pod_scaled_v2"))
    parser.add_argument("--ne-scale", type=float, default=1e19)
    parser.add_argument("--te-scale", type=float, default=1e3)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--variance-threshold", type=float, default=1e-6)
    args = parser.parse_args()
    if args.ne_scale <= 0 or args.te_scale <= 0 or args.batch_size < 1 or args.variance_threshold <= 0:
        parser.error("Scales, batch size, and variance threshold must be positive")
    prepare(args)


if __name__ == "__main__":
    main()
