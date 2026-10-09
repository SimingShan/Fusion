#!/usr/bin/env python3
"""Add input-only history to a frozen, already checked auxiliary diagnostic."""
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from fusion.data import arrow_to_numpy
from fusion.pod import read_manifest, split_entries, validate_splits
from fusion.thermal_history import extract, feature_names, analytic_check, OFFSETS_MS
from scripts.codabench_submit import write_json


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare_shot(task):
    import pyarrow.parquet as pq
    first, entry, relative_rows = task
    with np.load(entry["path"], allow_pickle=False) as values:
        query, frames = values["times"][relative_rows], values["frame_indices"][relative_rows]
    raw = ROOT / "data/raw" / entry["shard"]
    names = pq.ParquetFile(raw).schema_arrow.names
    allowed = {"source", "efit_times"} | {f"thomson_{s}_{f}" for s in ("core", "edge") for f in ("times", "Te", "ne")}
    columns = [name for name in names if name in allowed]
    table = pq.read_table(raw, columns=columns)
    row = {name: arrow_to_numpy(table[name][0]) for name in columns}
    if row["source"] != "DIII-D" or not np.array_equal(np.asarray(row["efit_times"])[frames], query):
        raise ValueError("Native input identity does not match the frozen feature rows")
    return first, extract(row, query)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, default=Path("results/evaluations/c1_probe_019"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--workers", type=int, default=2)
    a = p.parse_args()
    if a.output.exists():
        raise ValueError("Preserve existing prepared features")
    a.output.mkdir(parents=True)
    check = analytic_check()
    data = ROOT / "data/advanced_history_v2"
    validate_splits(data)
    manifest = read_manifest(data)
    original = json.loads((a.source / "run.json").read_text())
    train_hash = hashlib.sha256(json.dumps(split_entries(data, "train"), sort_keys=True).encode()).hexdigest()
    if original["status"] != "completed" or train_hash != original["source_training_entries_sha256"]:
        raise ValueError("The checked training feature contract changed")
    with np.load(a.source / "targets.npz", allow_pickle=False) as values:
        train_indices = values["train_global_indices"]
        counts = {"train": len(values["train"]), "val": len(values["val"])}
    if len(train_indices) != counts["train"] or np.any(np.diff(train_indices) <= 0):
        raise ValueError("Expected unique ordered training indices")
    parent_files = ["run.json", "train_features.npz", "val_features.npz", "targets.npz", "target_check.json"]
    report = {**original, "created_utc": datetime.now(timezone.utc).isoformat(), "status": "preparing",
              "source_diagnostic": str(a.source), "parent_sha256": {f: digest(a.source/f) for f in parent_files},
              "input_history": {"offsets_ms": OFFSETS_MS, "feature_names": feature_names(), "check": check,
                  "raw_columns": ["source", "efit_times", "thomson_core_times", "thomson_core_Te", "thomson_core_ne",
                                  "thomson_edge_times", "thomson_edge_Te", "thomson_edge_ne"]},
              "targets_reused_unchanged": True, "audit_labels_read": False, "test_labels_read": False,
              "eligible_for_submission": False}
    write_json(a.output / "run.json", report)
    sources = [Path(__file__), ROOT / "fusion/thermal_history.py", ROOT / "fusion/advanced_data.py",
               ROOT / "fusion/data.py", ROOT / "fusion/pod.py"]
    report["preparation_source_sha256"] = {}
    for path in sources:
        name = path.resolve().relative_to(ROOT)
        destination = a.output / "source_snapshot" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        report["preparation_source_sha256"][str(name)] = digest(path)
    for split in ("train", "val"):
        indices = train_indices if split == "train" else np.arange(counts[split])
        tasks, offset = [], 0
        for entry in manifest[split]:
            end = offset + entry["n_frames"]
            lo, hi = np.searchsorted(indices, [offset, end])
            if hi > lo:
                path = Path(entry["path"])
                chosen = dict(entry, path=str(path if path.is_absolute() else data/path))
                tasks.append((int(lo), chosen, indices[lo:hi]-offset))
            offset = end
        if indices[-1] >= offset or sum(len(task[2]) for task in tasks) != counts[split]:
            raise ValueError("Feature rows do not cover their declared split")
        temporary = a.output / f"{split}_history.npy"
        matrix = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float32,
                                          shape=(counts[split], len(feature_names())))
        completed = 0
        with ProcessPoolExecutor(max_workers=a.workers) as pool:
            for first, values in pool.map(prepare_shot, tasks, chunksize=4):
                matrix[first:first+len(values)] = values
                completed += len(values)
        if completed != counts[split]:
            raise ValueError("History preparation is incomplete")
        matrix.flush()
        with np.load(a.source / f"{split}_features.npz", allow_pickle=False) as values:
            old, baseline = values["X"], values["baseline"]
        if len(old) != len(matrix):
            raise ValueError("New inputs do not align with frozen predictions")
        if split == "train":
            active = np.flatnonzero(np.ptp(matrix, axis=0) > 1e-6)
            report["input_history"]["active_indices_from_training"] = active.tolist()
        combined = np.column_stack((old, matrix[:, active]))
        np.savez(a.output / f"{split}_features.npz", X=combined, baseline=baseline)
        del old, combined, baseline, matrix
        print(json.dumps({"split": split, "frames": completed, "added_nonconstant_features": len(active)}), flush=True)
    for name in ("targets.npz", "target_check.json"):
        shutil.copy2(a.source/name, a.output/name)
        if digest(a.output/name) != report["parent_sha256"][name]:
            raise ValueError("Frozen references changed while copying")
    report.update(status="completed", prepared_feature_sha256={f: digest(a.output/f) for f in
                  ("train_features.npz", "val_features.npz", "targets.npz")})
    write_json(a.output / "run.json", report)
    print(json.dumps({"complete": True, "train_frames": counts["train"], "val_frames": counts["val"],
                      "targets_reused_unchanged": True, "test_labels_read": False, "audit_labels_read": False}), flush=True)


if __name__ == "__main__":
    main()
