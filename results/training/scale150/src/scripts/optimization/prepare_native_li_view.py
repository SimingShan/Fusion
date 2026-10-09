#!/usr/bin/env python3
"""Remap unchanged training li references into a verified all-native cache."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from fusion.pod import load_arrays, manifest_hash, split_entries, validate_splits
from scripts.optimization.prepare_scaled_features import _link_or_copy


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            result.update(chunk)
    return result.hexdigest()


def entry_hash(entries):
    return hashlib.sha256(json.dumps(entries, sort_keys=True).encode()).hexdigest()


def remap_indices(old_offsets, new_offsets, old_native_indices, references):
    references = np.asarray(references, dtype=np.int64)
    if np.any(references < 0) or np.any(references >= old_offsets[-1]):
        raise ValueError("A source reference lies outside the original training rows")
    shot_positions = np.searchsorted(old_offsets, references, side="right") - 1
    native = np.asarray(old_native_indices)[references]
    if np.any(native < 0) or np.any(native >= np.diff(new_offsets)[shot_positions]):
        raise ValueError("A native reference lies outside the new shot")
    mapped = new_offsets[shot_positions] + native
    if len(np.unique(mapped)) != len(mapped):
        raise ValueError("Remapping introduced duplicate training references")
    return mapped


def prepare(source_li, data_dir, output):
    started = time.monotonic()
    source_li, data_dir, output = [Path(path).resolve() for path in (source_li, data_dir, output)]
    if output.exists():
        raise ValueError("Use a new target view directory; existing references are immutable")
    source = json.loads((source_li / "metadata.json").read_text())
    if not source.get("complete") or source["source_split"] != "train":
        raise ValueError("Only complete training references may be remapped")
    source_data = Path(source["source_data_dir"])
    validate_splits(source_data)
    validate_splits(data_dir)
    old, new = split_entries(source_data, "train"), split_entries(data_dir, "train")
    if entry_hash(old) != source["source_training_entries_sha256"] or len(old) != len(new):
        raise ValueError("Source target indexing or new shot count differs")
    old_values = load_arrays(source_data, "train", keys=("frame_indices", "times"))
    old_offsets = np.r_[0, np.cumsum([entry["n_frames"] for entry in old])]
    new_offsets = np.r_[0, np.cumsum([entry["n_frames"] for entry in new])]
    if old_offsets[-1] != source["total_training_frames"]:
        raise ValueError("Original training frame count differs from li references")
    R, Z = [np.load(source_li / f"{name}.npy", allow_pickle=False) for name in ("R", "Z")]
    # Check query identity for every old retained observation, reading only
    # native indices/times/grid scaffolding from the new shot caches.
    for i, (a, b) in enumerate(zip(old, new)):
        for key in ("shot_index", "shot_id", "shard", "row", "raw_frames"):
            if a.get(key) != b.get(key):
                raise ValueError(f"Training shot identity changed: {key}")
        if b["n_frames"] != b["raw_frames"]:
            raise ValueError("New target cache must contain every native observation")
        sl = slice(old_offsets[i], old_offsets[i+1])
        retained = old_values["frame_indices"][sl]
        with np.load(data_dir / b["path"], allow_pickle=False) as shot:
            if not np.array_equal(shot["frame_indices"], np.arange(b["raw_frames"])):
                raise ValueError("Full-native row order differs from native frame indices")
            if not np.array_equal(shot["times"][retained], old_values["times"][sl]):
                raise ValueError("A retained native observation changed its query time")
            if not np.array_equal(shot["R"], R) or not np.array_equal(shot["Z"], Z):
                raise ValueError("Native grid differs from the unchanged li functional")
        if (i+1) % 1000 == 0:
            print(f"Verified native li indexing: {i+1}/{len(new)} training shots", flush=True)
    references = np.load(source_li / "global_frame_indices.npy", allow_pickle=False)
    mapped = remap_indices(old_offsets, new_offsets, old_values["frame_indices"], references)
    output.mkdir(parents=True)
    source_hashes = {}
    for name in ("R", "Z", "valid", "inside_packed", "contours", "segment_lengths", "li"):
        path = source_li / f"{name}.npy"
        source_hashes[name] = digest(path)
        _link_or_copy(path, output / path.name)
        if digest(output / path.name) != source_hashes[name]:
            raise ValueError(f"A preserved li reference array changed: {name}")
    np.save(output / "global_frame_indices.npy", mapped, allow_pickle=False)
    metadata = {key: value for key, value in source.items() if key not in ("geometry", "reused_targets")}
    metadata.update(version=2, source_data_dir=str(data_dir), source_training_entries_sha256=entry_hash(new),
                    source_manifest_sha256=manifest_hash(data_dir), total_training_frames=int(new_offsets[-1]),
                    complete=True, auxiliary_targets_included=["li"],
                    remapping={"source_li": str(source_li), "source_metadata_sha256": digest(source_li / "metadata.json"),
                               "source_data_dir": str(source_data), "source_training_entries_sha256": entry_hash(old),
                               "training_shots_verified": len(new), "old_cached_observations_verified": int(old_offsets[-1]),
                               "new_native_frames": int(new_offsets[-1]), "reference_rows_unchanged": len(mapped),
                               "source_reference_array_sha256": source_hashes,
                               "old_global_indices_sha256": digest(source_li / "global_frame_indices.npy"),
                               "new_global_indices_sha256": digest(output / "global_frame_indices.npy"),
                               "target_values_recomputed": False, "labels_interpolated": False,
                               "new_cache_fields_read": ["frame_indices", "times", "R", "Z"],
                               "validation_labels_read": False, "audit_labels_read": False,
                               "code_sha256": digest(Path(__file__)), "completed_utc": datetime.now(timezone.utc).isoformat(),
                               "seconds": time.monotonic()-started})
    temporary = output / "metadata.partial.json"
    temporary.write_text(json.dumps(metadata, indent=2)+"\n")
    temporary.replace(output / "metadata.json")
    print(json.dumps(metadata["remapping"]), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-li", type=Path, default=Path("data/geometry_history_v4"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/advanced_history_train_native_v7"))
    parser.add_argument("--output", type=Path, default=Path("data/li_history_train_native_v7"))
    args = parser.parse_args()
    prepare(**vars(args))


if __name__ == "__main__":
    main()
