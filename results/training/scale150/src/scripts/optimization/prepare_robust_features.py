#!/usr/bin/env python3
"""Keep availability on its native 0--1 scale and preserve dropout indicators."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from fusion.pod import load_arrays, manifest_hash, read_manifest, standardize_x, validate_splits
from scripts.optimization.prepare_scaled_features import fit_statistics, _link_or_copy


def prepare(data_dir, source_pod, output, batch_size=8192):
    started = time.monotonic()
    data_dir, source_pod, output = [Path(p).resolve() for p in (data_dir, source_pod, output)]
    if source_pod == output:
        raise ValueError("Robust normalization requires a separate output directory")
    validate_splits(data_dir)
    manifest = read_manifest(data_dir)
    names = manifest["feature_names"]
    source_meta = json.loads((source_pod / "basis.json").read_text())
    if source_meta["source_manifest_sha256"] != manifest_hash(data_dir):
        raise ValueError("Source POD and input cache provenance disagree")
    with np.load(source_pod / "normalization.npz", allow_pickle=False) as saved:
        norm = {key: saved[key].copy() for key in saved.files}
    X = load_arrays(data_dir, "train", keys=("X",))["X"]
    scales = norm.get("raw_log_scale", np.ones(len(names)))
    mean, std = fit_statistics(X, scales, batch_size)
    masks = np.asarray(["observed" in name or name.endswith("_coverage") for name in names])
    dropout_masks = np.asarray([name.startswith(("thomson_core_Te_", "thomson_core_ne_", "thomson_edge_Te_", "thomson_edge_ne_")) for name in names]) & masks
    active = np.flatnonzero((std > 1e-6) | (dropout_masks & (mean > 0)))
    # Constant positive Thomson masks become variable under modality dropout.
    # Other constant features remain excluded to avoid redundant dimensions.
    mean[masks], std[masks] = 0, 1
    norm.update(feature_indices=active, x_mean=mean[active].astype(np.float32), x_std=std[active].astype(np.float32))
    output.mkdir(parents=True, exist_ok=True)
    temporary = output / "normalization.tmp"
    with temporary.open("wb") as handle:
        np.savez(handle, **norm)
    os.replace(temporary, output / "normalization.npz")
    for split in ("train", "val", "legacy_val"):
        raw = X if split == "train" else load_arrays(data_dir, split, keys=("X",))["X"]
        temporary = output / f"{split}_X.partial.npy"
        normalized = np.lib.format.open_memmap(temporary, mode="w+", dtype="float32", shape=(len(raw), len(active)))
        for first in range(0, len(raw), batch_size):
            normalized[first:first + batch_size] = standardize_x(raw[first:first + batch_size], norm)
        normalized.flush()
        del normalized
        os.replace(temporary, output / f"{split}_X.npy")
        for key in ("coefficients", "scalars"):
            _link_or_copy(source_pod / f"{split}_{key}.npy", output / f"{split}_{key}.npy")
        print(f"Robust features {split}: {len(raw)} frames", flush=True)
    if (source_pod / "basis_sample_indices.npy").exists():
        _link_or_copy(source_pod / "basis_sample_indices.npy", output / "basis_sample_indices.npy")
    result = {**source_meta, "active_features": len(active),
              "robust_input": {"availability_normalization": "raw 0--1 values; mean=0, std=1",
                               "positive_constant_thomson_masks_retained": True,
                               "fit_split": "train", "basis_and_targets_reused": True,
                               "source_pod": str(source_pod), "audit_labels_read": False,
                               "source_normalization_sha256": hashlib.sha256((source_pod / "normalization.npz").read_bytes()).hexdigest(),
                               "active_availability_features": int(masks[active].sum()),
                               "seconds": time.monotonic() - started}}
    temporary = output / "basis.tmp.json"
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    os.replace(temporary, output / "basis.json")
    print(json.dumps({"active_features": len(active), "seconds": time.monotonic() - started}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/advanced_history_v2"))
    parser.add_argument("--source-pod", type=Path, default=Path("data/pod_history64_v2"))
    parser.add_argument("--output", type=Path, default=Path("data/pod_history64_robust_v3"))
    parser.add_argument("--batch-size", type=int, default=8192)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("Batch size must be positive")
    prepare(**vars(args))


if __name__ == "__main__":
    main()
