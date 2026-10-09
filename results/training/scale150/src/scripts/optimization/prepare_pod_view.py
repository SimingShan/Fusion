#!/usr/bin/env python3
"""Select existing normalized features and leading POD modes without refitting."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from fusion.pod import manifest_hash, read_manifest, split_entries, validate_splits


def _atomic_json(path, value):
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def _reuse(source, destination):
    if destination.exists() and os.path.samefile(source, destination):
        return
    temporary = destination.with_name(destination.name + f".{os.getpid()}.tmp")
    if temporary.exists():
        temporary.unlink()
    try:
        os.link(source, temporary)
    except OSError:
        shutil.copy2(source, temporary)
    os.replace(temporary, destination)


def prepare_view(data_dir, source_pod, output, rank=64, feature_group="currents", batch_size=8192):
    data_dir, source_pod, output = map(lambda p: Path(p).resolve(), (data_dir, source_pod, output))
    if source_pod == output:
        raise ValueError("A POD view must use a separate directory")
    validate_splits(data_dir)
    manifest = read_manifest(data_dir)
    names = manifest["feature_names"]
    metadata = json.loads((source_pod / "basis.json").read_text())
    if metadata["source_manifest_sha256"] != manifest_hash(data_dir):
        raise ValueError("Source POD provenance disagrees with the prepared data")
    with np.load(source_pod / "normalization.npz", allow_pickle=False) as archive:
        norm = {key: archive[key].copy() for key in archive.files}
    source_rank = len(norm["basis"])
    if not 1 <= rank <= source_rank or batch_size < 1:
        raise ValueError("Rank must fit within the source basis, and batch size must be positive")
    original_indices = norm["feature_indices"].copy()
    if feature_group == "currents":
        columns = np.asarray([i for i, raw in enumerate(original_indices)
                              if names[raw].startswith("magnetics_") or names[raw] == "time_seconds"], dtype=np.int64)
    elif feature_group == "all":
        columns = np.arange(len(original_indices), dtype=np.int64)
    else:
        raise ValueError(f"Unknown feature group {feature_group}")
    if not len(columns):
        raise ValueError("No active features match the requested group")
    fingerprint = hashlib.sha256(json.dumps({"basis_sha256": hashlib.sha256((source_pod / "basis.json").read_bytes()).hexdigest(),
                                             "normalization_sha256": hashlib.sha256((source_pod / "normalization.npz").read_bytes()).hexdigest(),
                                             "rank": rank, "feature_group": feature_group,
                                             "source_columns": columns.tolist()}, sort_keys=True).encode()).hexdigest()
    if (output / "basis.json").exists():
        previous = json.loads((output / "basis.json").read_text())
        if previous.get("pod_view", {}).get("fingerprint") != fingerprint:
            raise ValueError("Existing view has different settings/source; use a new output directory")
        if previous.get("pod_view", {}).get("complete"):
            return previous
    output.mkdir(parents=True, exist_ok=True)
    retained_fraction = float(norm["eigenvalues"][:rank].sum() / norm["eigenvalues"].sum())
    for key in ("basis", "coefficient_std", "eigenvalues"):
        norm[key] = norm[key][:rank].copy()
    for key in ("feature_indices", "x_mean", "x_std"):
        norm[key] = norm[key][columns].copy()
    temporary = output / f"normalization.{os.getpid()}.tmp"
    with temporary.open("wb") as handle:
        np.savez(handle, **norm)
    os.replace(temporary, output / "normalization.npz")
    split_frames = {}
    for split in ("train", "val", "legacy_val"):
        entries = split_entries(data_dir, split)
        if not entries:
            continue
        n = sum(entry["n_frames"] for entry in entries)
        for key, source_width, selection in (("X", len(original_indices), columns),
                                              ("coefficients", source_rank, slice(0, rank)),
                                              ("scalars", 2, slice(None))):
            source = source_pod / f"{split}_{key}.npy"
            array = np.load(source, mmap_mode="r", allow_pickle=False)
            if array.shape != (n, source_width) or array.dtype != np.float32:
                raise ValueError(f"Source dimensions/dtype disagree for {source}")
            width = len(columns) if key == "X" else rank if key == "coefficients" else 2
            if width == source_width:
                _reuse(source, output / source.name)
                continue
            temporary = output / f"{split}_{key}.partial.npy"
            target = np.lib.format.open_memmap(temporary, mode="w+", dtype="float32", shape=(n, width))
            for first in range(0, n, batch_size):
                target[first:first + batch_size] = array[first:first + batch_size, selection]
            target.flush()
            del target
            os.replace(temporary, output / source.name)
        split_frames[split] = n
        print(f"Prepared {split} view: {n} frames, {len(columns)} features, rank {rank}", flush=True)
    if (source_pod / "basis_sample_indices.npy").exists():
        _reuse(source_pod / "basis_sample_indices.npy", output / "basis_sample_indices.npy")
    result = {**metadata, "rank": rank, "active_features": len(columns),
              "basis_variance_fraction": metadata["basis_variance_fraction"] * retained_fraction,
              "pod_view": {"feature_group": feature_group, "rank": rank, "source_rank": source_rank,
                           "source_pod": str(source_pod), "source_normalized_columns": columns.tolist(),
                           "raw_feature_indices": norm["feature_indices"].tolist(),
                           "selected_feature_names": [names[i] for i in norm["feature_indices"]],
                           "basis_refitted": False, "normalization_refitted": False,
                           "scalar_targets_unchanged": True, "split_frames": split_frames,
                           "fingerprint": fingerprint, "complete": True}}
    _atomic_json(output / "basis.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/advanced_v1"))
    parser.add_argument("--source-pod", type=Path, default=Path("data/pod_full_v1"))
    parser.add_argument("--output", type=Path, default=Path("data/pod_currents64_v1"))
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--feature-group", choices=("currents", "all"), default="currents")
    parser.add_argument("--batch-size", type=int, default=8192)
    result = prepare_view(**vars(parser.parse_args()))
    print(json.dumps({"rank": result["rank"], "active_features": result["active_features"],
                      "view": result["pod_view"]["feature_group"]}), flush=True)


if __name__ == "__main__":
    main()
