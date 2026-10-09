"""Training loss weights representing counts of original native frames."""
from __future__ import annotations

import hashlib
import json
import numpy as np

from .pod import load_arrays, split_entries


def nearest_native_counts(retained_indices, raw_frames):
    """Count nearest retained-index assignments; exact midpoint ties go left.

    This counts discrete native observations. It does not integrate elapsed
    time, interpolate labels, or read any output target values.
    """
    retained = np.asarray(retained_indices)
    if retained.ndim != 1 or not len(retained) or not np.issubdtype(retained.dtype, np.integer):
        raise ValueError("Retained native indices must be a nonempty integer vector")
    if raw_frames < 1 or retained[0] < 0 or retained[-1] >= raw_frames or np.any(np.diff(retained) <= 0):
        raise ValueError("Retained indices must be unique, ordered, and within the native frame range")
    retained = retained.astype(np.int64)
    boundaries = (retained[:-1] + retained[1:]) // 2 + 1
    counts = np.diff(np.r_[0, boundaries, int(raw_frames)]).astype(np.int64)
    if np.any(counts <= 0) or counts.sum() != raw_frames:
        raise ValueError("Nearest-index partition must cover every original native frame exactly once")
    return counts


def training_native_weights(data_dir):
    entries = split_entries(data_dir, "train")
    native_indices = load_arrays(data_dir, "train", keys=("frame_indices",))["frame_indices"]
    counts = np.empty(len(native_indices), dtype=np.int64)
    offset = 0
    for entry in entries:
        stop = offset + entry["n_frames"]
        counts[offset:stop] = nearest_native_counts(native_indices[offset:stop], entry["raw_frames"])
        offset = stop
    if offset != len(counts):
        raise ValueError("Training entries do not cover the consolidated native indices")
    weights = (counts / counts.mean()).astype(np.float32)
    metadata = {
        "method": "discrete_native_index_nearest_retained", "tie_rule": "earlier retained native index",
        "fit_split": "train", "training_shots": len(entries), "cached_frames": len(counts),
        "represented_native_frames": int(counts.sum()), "count_range": [int(counts.min()), int(counts.max())],
        "normalization": "counts divided by global training count mean; loss uses batch weighted mean",
        "target_normalization_changed": False, "labels_interpolated": False,
        "source_training_entries_sha256": hashlib.sha256(json.dumps(entries, sort_keys=True).encode()).hexdigest(),
        "native_indices_sha256": hashlib.sha256(np.asarray(native_indices).tobytes()).hexdigest(),
        "counts_sha256": hashlib.sha256(counts.tobytes()).hexdigest(),
        "target_values_read": False, "validation_labels_read": False, "audit_labels_read": False,
    }
    return weights, counts, metadata
