#!/usr/bin/env python3
"""Expand the existing training shots to native frames with frozen POD statistics.

This prepares data only. All labels opened belong to the existing training split.
Validation/legacy compact arrays are reused byte-for-byte; audit entries are omitted.
No basis, normalization, model, or target interpolation is fitted by this script.
"""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import datetime, timezone
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
from fusion.advanced_data import prepare_advanced_dataset
from fusion.pod import manifest_hash, standardize_x, validate_splits
from scripts.optimization.prepare_history_features import _task as history_task, history_feature_names, LAGS_MS


def utc():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temp, path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def array_digest(array):
    array = np.ascontiguousarray(array)
    h = hashlib.sha256(json.dumps([str(array.dtype), list(array.shape)]).encode())
    h.update(memoryview(array).cast("B"))
    return h.hexdigest()


def copy_frozen(source, destination):
    source, destination = Path(source), Path(destination)
    if destination.exists():
        if digest(source) != digest(destination):
            raise ValueError(f"Frozen copy differs: {destination}")
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    return digest(destination)


def link_frozen(source, destination):
    source, destination = Path(source), Path(destination)
    if not destination.exists():
        try:
            os.link(source, destination)
        except OSError:
            shutil.copy2(source, destination)
    if source.stat().st_size != destination.stat().st_size or digest(source) != digest(destination):
        raise ValueError(f"Reused development array differs: {destination}")
    return {"source": str(source), "path": str(destination), "sha256": digest(destination),
            "bytes": destination.stat().st_size,
            "same_inode": source.stat().st_ino == destination.stat().st_ino and source.stat().st_dev == destination.stat().st_dev}


def config_and_plan(args):
    source = json.loads((args.source_view / "manifest.json").read_text())
    basis = json.loads((args.source_pod / "basis.json").read_text())
    if basis["source_manifest_sha256"] != manifest_hash(args.source_view):
        raise ValueError("Frozen source POD does not match the source history view")
    original = json.loads((args.source_base / "manifest.json").read_text())
    if source["feature_names"] != original["feature_names"] + history_feature_names():
        raise ValueError("The current input/history feature order differs from the source")
    for relative, expected in original["settings"]["source_sha256"].items():
        if digest(ROOT / relative) != expected:
            raise ValueError(f"Base feature source has changed since the source cache: {relative}")
    expected_shots = [int(e["shot_index"]) for e in source["train"]]
    if expected_shots != [int(e["shot_index"]) for e in original["train"]]:
        raise ValueError("Base and history training shot order differ")
    plan = prepare_advanced_dataset(raw_dir=args.raw_dir, output=args.base_output,
                                   legacy_dir=args.legacy_dir, train_frames=0,
                                   val_frames=original["settings"]["val_frames"],
                                   seed=original["settings"]["seed"],
                                   val_shots=original["settings"]["val_shots"],
                                   audit_shots=original["settings"]["audit_shots"],
                                   splits=["train"], workers=args.workers,
                                   consolidate=False, plan_only=True)
    if plan["selection"]["train"] != expected_shots or plan["dataset"] != source["dataset"]:
        raise ValueError("Native preparation would change shot identities or dataset revision")
    frames = sum(int(e["raw_frames"]) for e in source["train"])
    cached = sum(int(e["n_frames"]) for e in source["train"])
    if (len(expected_shots), frames, cached) != (6753, 1496407, 426684):
        raise ValueError("Unexpected source training identities/counts for this controlled expansion")
    sources = ["fusion/advanced_data.py", "fusion/data.py", "fusion/pod.py",
               "starter/data_fixes.py", "scripts/optimization/prepare_history_features.py",
               "scripts/optimization/prepare_native_training.py"]
    configuration = {
        "source_base": str(args.source_base), "source_base_manifest_sha256": manifest_hash(args.source_base),
        "source_view": str(args.source_view), "source_view_manifest_sha256": manifest_hash(args.source_view),
        "source_pod": str(args.source_pod), "source_pod_basis_sha256": digest(args.source_pod / "basis.json"),
        "normalization_sha256": digest(args.source_pod / "normalization.npz"),
        "base_output": str(args.base_output), "history_output": str(args.history_output),
        "pod_output": str(args.pod_output), "raw_dir": str(args.raw_dir),
        "train_shots": len(expected_shots), "training_frames": frames,
        "source_training_frames": cached, "dataset": source["dataset"],
        "train_shot_indices_sha256": array_digest(np.asarray(expected_shots, dtype=np.int64)),
        "source_code_sha256": {name: digest(ROOT / name) for name in sources},
        "feature_count": len(source["feature_names"]), "active_features": basis["active_features"],
        "rank": basis["rank"], "train_frame_cap": 0,
        "normalization_refitted": False, "basis_refitted": False,
        "labels_read_splits": ["train"], "audit_labels_read": False,
        "development_frames_changed": False, "target_interpolation": False,
        "compact_row_policy": {"original_rows": "reuse_bitwise", "original_row_count": cached,
                               "new_row_count": frames - cached,
                               "new_coefficient_projection_dtype": "float32",
                               "new_projection_batch_size": args.batch_size,
                               "projection_batches_cross_shots": False},
    }
    widths = {"base_psi_float32": 4225, "base_X_float32": len(original["feature_names"]),
              "base_scalars_float32": 2, "history_sidecars_float32": len(history_feature_names()),
              "pod_X_float32": basis["active_features"], "pod_coefficients_float32": basis["rank"],
              "pod_scalars_float32": 2}
    byte_estimates = {key: frames * width * 4 for key, width in widths.items()}
    byte_estimates["base_times_and_frame_indices"] = frames * 16
    byte_estimates["compact_times_frame_indices_shot_indices"] = frames * 24
    available = os.statvfs(args.results)
    estimate = {"array_bytes": byte_estimates, "total_GB": sum(byte_estimates.values()) / 1e9,
                "total_GiB": sum(byte_estimates.values()) / 2**30,
                "filesystem_free_GiB": available.f_bavail * available.f_frsize / 2**30,
                "per_user_quota_verified": False,
                "ram_request_GiB": 32, "workers": args.workers,
                "runtime_estimate_minutes": [10, 20],
                "note": "Metadata/ZIP headers add a small overhead; no duplicate consolidated raw X or psi."}
    return configuration, estimate, source, original, basis


def assert_sources(configuration):
    for relative, expected in configuration["source_code_sha256"].items():
        if digest(ROOT / relative) != expected:
            raise ValueError(f"Preparation source changed during this run: {relative}")
    for key, filename in [("source_base", "manifest.json"), ("source_view", "manifest.json"),
                          ("source_pod", "normalization.npz")]:
        expected = configuration["normalization_sha256"] if key == "source_pod" else configuration[key + "_manifest_sha256"]
        if digest(Path(configuration[key]) / filename) != expected:
            raise ValueError(f"Frozen source changed during preparation: {key}")


def prepare_history(args, base, source, config_sha):
    path = args.history_output / "manifest.json"
    if path.exists():
        view = json.loads(path.read_text())
        if view["native_training_expansion"]["configuration_sha256"] != config_sha:
            raise ValueError("Existing history output has a different frozen configuration")
        if view.get("complete_splits", {}).get("train"):
            return view
    else:
        view = json.loads(json.dumps(base))
        view["feature_names"] = source["feature_names"]
        view["consolidated"] = {}
        view["audit"], view["test"] = [], []
        view["complete_splits"] = {"train": False, "val": True, "legacy_val": True, "audit": False, "test": False}
        for entry in view["train"]:
            entry["path"] = str((args.base_output / entry["path"]).resolve())
            entry["X_extra"] = f"history/train/shot_{entry['shot_index']:05d}.npy"
        for split in ("val", "legacy_val"):
            view[split] = json.loads(json.dumps(source[split]))
            for entry in view[split]:
                entry["path"] = str((args.source_view / entry["path"]).resolve())
                entry["X_extra"] = str((args.source_view / entry["X_extra"]).resolve())
        view["feature_augmentation"] = {
            "configuration": {"base_dir": str(args.base_output), "base_manifest_sha256": manifest_hash(args.base_output),
                              "lags_ms": list(LAGS_MS),
                              "source_code_sha256": digest(ROOT / "scripts/optimization/prepare_history_features.py")},
            "base_feature_count": len(base["feature_names"]), "extra_feature_count": 210,
            "complete_splits": {"train": False, "val": True, "legacy_val": True},
            "target_values_read": False}
        view["native_training_expansion"] = {
            "configuration_sha256": config_sha, "source_view": str(args.source_view),
            "training_native_frames_only": True, "training_frame_cap": 0,
            "validation_and_legacy_entries_reused": True,
            "audit_entries_omitted": True, "audit_labels_read": False,
            "targets_interpolated": False, "normalization_refitted": False, "basis_refitted": False}
        write_json(path, view)
    tasks = [{"output": str(args.history_output), "raw_dir": str(args.raw_dir), "split": "train", "entry": entry}
             for entry in view["train"]]
    started = time.monotonic()
    completed = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        iterator = iter(tasks)
        pending = {pool.submit(history_task, task) for task in [next(iterator) for _ in range(min(len(tasks), args.workers * 2))]}
        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                future.result()
                completed += 1
                task = next(iterator, None)
                if task is not None:
                    pending.add(pool.submit(history_task, task))
                if completed % 256 == 0:
                    print(f"History {completed}/{len(tasks)} shots, {time.monotonic()-started:.1f}s", flush=True)
    view["complete_splits"]["train"] = True
    view["feature_augmentation"]["complete_splits"]["train"] = True
    write_json(path, view)
    validate_splits(args.history_output)
    return view


def project_training(args, view, source, norm, config_sha):
    from threadpoolctl import threadpool_limits

    entries = view["train"]
    n = sum(e["n_frames"] for e in entries)
    state_path = args.results / "projection_state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {
        "configuration_sha256": config_sha, "completed_shots": 0, "completed_frames": 0,
        "original_frames_verified": 0, "records": [], "complete": False}
    if state["configuration_sha256"] != config_sha:
        raise ValueError("Projection resume configuration differs")
    if state["complete"]:
        return state
    specs = {"X": ((n, len(norm["feature_indices"])), "float32"),
             "coefficients": ((n, len(norm["basis"])), "float32"),
             "scalars": ((n, 2), "float32"), "times": ((n,), "float64"),
             "frame_indices": ((n,), "int64"), "shot_index": ((n,), "int64")}
    maps = {}
    for key, (shape, dtype) in specs.items():
        path = args.pod_output / f"train_{key}.partial.npy"
        final = args.pod_output / f"train_{key}.npy"
        if state["completed_shots"] == len(entries) and final.exists() and not path.exists():
            maps[key] = np.load(final, mmap_mode="r+", allow_pickle=False)
        elif path.exists():
            maps[key] = np.load(path, mmap_mode="r+", allow_pickle=False)
        elif state["completed_shots"]:
            raise ValueError(f"Missing partially projected array: {path}")
        else:
            maps[key] = np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)
        if maps[key].shape != shape or maps[key].dtype != np.dtype(dtype):
            raise ValueError(f"Unexpected projection shape/dtype: {path}")
    source_offsets = np.cumsum([0] + [e["n_frames"] for e in source["train"]])
    source_compact = {key: np.load(args.source_pod / f"train_{key}.npy", mmap_mode="r", allow_pickle=False)
                      for key in ("X", "coefficients", "scalars")}
    start_time = time.monotonic()
    maxima = state.setdefault("reused_compact_row_max_abs_difference", {key: 0.0 for key in source_compact})
    with threadpool_limits(limits=args.workers, user_api="blas"):
        for number in range(state["completed_shots"], len(entries)):
            entry, old_entry = entries[number], source["train"][number]
            if (entry["shot_index"], entry["shard"], entry["row"], entry["shot_id"]) != (old_entry["shot_index"], old_entry["shard"], old_entry["row"], old_entry["shot_id"]):
                raise ValueError("Shot identity differs from frozen training split")
            with np.load(entry["path"], allow_pickle=False) as native:
                arrays = {key: native[key] for key in ("X", "psi", "scalars", "times", "frame_indices", "R", "Z")}
            frames = np.arange(entry["raw_frames"], dtype=np.int64)
            if entry["n_frames"] != entry["raw_frames"] or not np.array_equal(arrays["frame_indices"], frames):
                raise ValueError("Native training cache omitted/reordered target frames")
            if not np.isfinite(arrays["psi"]).all() or not np.isfinite(arrays["times"]).all():
                raise ValueError("Invalid native flux or timestamp")
            extra = np.load(args.history_output / entry["X_extra"], allow_pickle=False)
            combined = np.concatenate((arrays["X"], extra), axis=1)
            if combined.shape != (len(frames), len(view["feature_names"])) or not np.isfinite(combined).all():
                raise ValueError("Invalid full native input features")
            offset = state["completed_frames"]
            end = offset + len(frames)
            with np.load(args.source_view / old_entry["path"], allow_pickle=False) as old_query:
                old_frames = old_query["frame_indices"]
            # Preserve the original float32 BLAS outputs exactly. A different
            # batch shape can perturb tiny high-order modes despite equal psi.
            # Only additional native observations need a new projection.
            for key, original in source_compact.items():
                maps[key][offset + old_frames] = original[source_offsets[number]:source_offsets[number + 1]]
            new_frames = np.setdiff1d(frames, old_frames, assume_unique=True)
            for first in range(0, len(new_frames), args.batch_size):
                selected = new_frames[first:first + args.batch_size]
                block = offset + selected
                maps["X"][block] = standardize_x(combined[selected], norm)
                centered = arrays["psi"][selected].reshape(-1, 4225) - norm["psi_mean"].reshape(-1)
                maps["coefficients"][block] = (centered @ norm["basis"].T) / norm["coefficient_std"]
                maps["scalars"][block] = (arrays["scalars"][selected] - norm["scalar_mean"]) / norm["scalar_std"]
            maps["times"][offset:end] = arrays["times"]
            maps["frame_indices"][offset:end] = frames
            maps["shot_index"][offset:end] = entry["shot_index"]
            with np.load(args.source_view / old_entry["path"], allow_pickle=False) as old:
                old_frames = old["frame_indices"]
                for key in ("X", "psi", "scalars", "times"):
                    if not np.array_equal(arrays[key][old_frames], old[key], equal_nan=True):
                        raise ValueError(f"Previously cached training {key} changed on shot {entry['shot_index']}")
                for key in ("R", "Z"):
                    if not np.array_equal(arrays[key], old[key], equal_nan=True):
                        raise ValueError(f"Native physical grid {key} changed")
            old_extra = np.load(args.source_view / old_entry["X_extra"], allow_pickle=False)
            if not np.array_equal(extra[old_frames], old_extra):
                raise ValueError("Previously cached history values changed")
            for key, original in source_compact.items():
                reference = original[source_offsets[number]:source_offsets[number + 1]]
                actual = np.asarray(maps[key][offset + old_frames])
                finite = np.isfinite(reference) & np.isfinite(actual)
                error = float(np.max(np.abs(actual[finite] - reference[finite]))) if finite.any() else 0.0
                maxima[key] = max(maxima[key], error)
                if not np.array_equal(actual, reference, equal_nan=True):
                    raise ValueError(f"Reused compact {key} rows are not bitwise equal to the source")
                if key == "X" and not np.array_equal(standardize_x(combined[old_frames], norm), reference, equal_nan=True):
                    raise ValueError("Frozen input normalization no longer reproduces source rows")
                if key == "scalars" and not np.array_equal((arrays["scalars"][old_frames] - norm["scalar_mean"]) / norm["scalar_std"], reference, equal_nan=True):
                    raise ValueError("Frozen scalar normalization no longer reproduces source rows")
            record = {"shot_index": entry["shot_index"], "shot_id": entry["shot_id"],
                      "shard": entry["shard"], "row": entry["row"], "offset": offset, "n_frames": len(frames),
                      "source_n_frames": len(old_frames),
                      "array_sha256": {key: array_digest(value) for key, value in arrays.items()},
                      "history_sha256": array_digest(extra), "original_cached_frames_bitwise_verified": True,
                      "original_compact_rows_reused_bitwise": len(old_frames),
                      "newly_projected_native_rows": len(new_frames)}
            state["records"].append(record)
            state["completed_shots"], state["completed_frames"] = number + 1, end
            state["original_frames_verified"] += len(old_frames)
            if (number + 1) % 64 == 0 or number + 1 == len(entries):
                for value in maps.values():
                    value.flush()
                state["updated_at_utc"] = utc()
                write_json(state_path, state)
            if (number + 1) % 256 == 0:
                print(f"Projected and verified {number+1}/{len(entries)} shots, {end}/{n} frames, {time.monotonic()-start_time:.1f}s", flush=True)
    for value in maps.values():
        value.flush()
    maps.clear()
    for key in specs:
        temp = args.pod_output / f"train_{key}.partial.npy"
        if temp.exists():
            os.replace(temp, args.pod_output / f"train_{key}.npy")
    np.save(args.pod_output / "train_offsets.npy", np.cumsum([0] + [e["n_frames"] for e in entries], dtype=np.int64))
    state["complete"] = True
    write_json(state_path, state)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag, default in [("raw-dir", "data/raw"), ("legacy-dir", "data/processed"),
                          ("source-base", "data/advanced_v1"), ("source-view", "data/advanced_history_v2"),
                          ("source-pod", "data/pod_history128_robust_v4"),
                          ("base-output", "data/advanced_train_native_v7"),
                          ("history-output", "data/advanced_history_train_native_v7"),
                          ("pod-output", "data/pod_history128_train_native_v7"),
                          ("results", "results/optimization/native_training_preparation_v7")]:
        parser.add_argument("--" + flag, type=Path, default=Path(default))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    if args.workers < 1 or args.batch_size < 1:
        parser.error("workers and batch size must be positive")
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.resolve())
    outputs = [args.base_output, args.history_output, args.pod_output]
    if len(set(outputs)) != 3 or any(p in (args.source_base, args.source_view, args.source_pod) for p in outputs):
        raise ValueError("Use three separate new cache directories")
    args.results.mkdir(parents=True, exist_ok=True)
    configuration, estimate, source, original, source_basis = config_and_plan(args)
    config_sha = hashlib.sha256(json.dumps(configuration, sort_keys=True).encode()).hexdigest()
    write_json(args.results / "plan.json", {"prepared_at_utc": utc(), "configuration": configuration,
                                           "configuration_sha256": config_sha, "storage_estimate": estimate})
    print(json.dumps({"configuration_sha256": config_sha, "storage_estimate": estimate,
                      "frames": configuration["training_frames"], "shots": configuration["train_shots"],
                      "plan_only": args.plan_only}), flush=True)
    if args.plan_only:
        return
    if (args.results / "summary.json").exists():
        saved = json.loads((args.results / "summary.json").read_text())
        if saved["configuration_sha256"] != config_sha:
            raise ValueError("Completed output has a different configuration")
        print("Preparation already completed and configuration matches; no labels opened.", flush=True)
        return
    protocol_path = args.results / "protocol.json"
    if protocol_path.exists():
        if json.loads(protocol_path.read_text())["configuration_sha256"] != config_sha:
            raise ValueError("Existing preparation protocol differs")
    else:
        write_json(protocol_path, {"registered_at_utc": utc(), "configuration": configuration,
                                   "configuration_sha256": config_sha,
                                   "purpose": "Controlled increase of training time density only; not an audit or model trial."})
    started = time.monotonic()
    args.pod_output.mkdir(parents=True, exist_ok=True)
    args.history_output.mkdir(parents=True, exist_ok=True)
    provenance = args.results / "provenance"
    for relative in configuration["source_code_sha256"]:
        copy_frozen(ROOT / relative, provenance / "source" / relative)
    for path, name in [(args.source_pod / "basis.json", "source_basis.json"),
                       (args.source_view / "manifest.json", "source_history_manifest.json"),
                       (args.source_base / "manifest.json", "source_base_manifest.json")]:
        copy_frozen(path, provenance / name)
    copy_frozen(args.source_pod / "normalization.npz", args.pod_output / "normalization.npz")
    assert_sources(configuration)
    print("Preparing native training base; all other target splits excluded.", flush=True)
    base = prepare_advanced_dataset(raw_dir=args.raw_dir, output=args.base_output, legacy_dir=args.legacy_dir,
                                    seed=original["settings"]["seed"], val_shots=original["settings"]["val_shots"],
                                    audit_shots=original["settings"]["audit_shots"], train_frames=0,
                                    val_frames=original["settings"]["val_frames"], splits=["train"],
                                    workers=args.workers, consolidate=False)
    if len(base["train"]) != configuration["train_shots"] or sum(e["n_frames"] for e in base["train"]) != configuration["training_frames"]:
        raise ValueError("Native preparation produced unexpected shot/frame counts")
    if any(base.get(split) for split in ("val", "legacy_val", "audit", "test")):
        raise ValueError("Native base cache unexpectedly includes an unrequested split")
    view = prepare_history(args, base, source, config_sha)
    with np.load(args.pod_output / "normalization.npz", allow_pickle=False) as archive:
        norm = {key: archive[key].copy() for key in archive.files}
    projection = project_training(args, view, source, norm, config_sha)
    reused = {}
    for split in ("val", "legacy_val"):
        for key in ("X", "coefficients", "scalars"):
            name = f"{split}_{key}.npy"
            reused[name] = link_frozen(args.source_pod / name, args.pod_output / name)
    copy_frozen(args.source_pod / "basis_sample_indices.npy", provenance / "source_basis_sample_indices.npy")
    metadata = {**source_basis, "source_manifest_sha256": manifest_hash(args.history_output),
                "data_dir": str(args.history_output), "training_frames": configuration["training_frames"],
                "basis_fit_training_frames": source_basis["training_frames"],
                "basis_sample_indices_scope": "Original 426684-frame source cache only; indices are not native-cache indices.",
                "source_basis_sample_indices_path": str(provenance / "source_basis_sample_indices.npy"),
                "native_training_expansion": {**configuration, "configuration_sha256": config_sha,
                                               "source_basis_metadata": str(provenance / "source_basis.json"),
                                               "reused_development_arrays": reused,
                                               "old_li_geometry_stores_compatible": False,
                                               "native_count_weighting_needed": False}}
    # Deliberately do not place old basis-sample indices beside new training arrays:
    # their flattened offsets refer to the old sampling, not the expanded cache.
    write_json(args.pod_output / "basis.json", metadata)
    assert_sources(configuration)
    if digest(args.pod_output / "normalization.npz") != configuration["normalization_sha256"]:
        raise ValueError("Frozen normalization changed")
    validate_splits(args.history_output)
    artifacts = {}
    for path in sorted(args.pod_output.glob("*.npy")):
        artifacts[path.name] = {"bytes": path.stat().st_size, "sha256": digest(path)}
    summary = {"completed_at_utc": utc(), "configuration_sha256": config_sha,
               "elapsed_seconds_this_attempt": time.monotonic() - started,
               "training_shots": projection["completed_shots"], "training_frames": projection["completed_frames"],
               "original_training_frames_bitwise_verified": projection["original_frames_verified"],
               "density_multiplier": configuration["training_frames"] / configuration["source_training_frames"],
               "input_features": len(view["feature_names"]), "active_features": len(norm["feature_indices"]),
               "rank": len(norm["basis"]), "normalization_sha256": configuration["normalization_sha256"],
               "reused_compact_row_max_abs_difference": projection["reused_compact_row_max_abs_difference"],
               "original_compact_rows_reused_bitwise": projection["original_frames_verified"],
               "newly_projected_native_rows": projection["completed_frames"] - projection["original_frames_verified"],
               "compact_row_policy": configuration["compact_row_policy"],
               "base_manifest_sha256": manifest_hash(args.base_output),
               "history_manifest_sha256": manifest_hash(args.history_output),
               "basis_metadata_sha256": digest(args.pod_output / "basis.json"),
               "projection_records_sha256": digest(args.results / "projection_state.json"),
               "basis_refitted": False, "normalization_refitted": False, "target_interpolation": False,
               "audit_labels_read": False, "labels_read_splits": ["train"],
               "validation_and_legacy_arrays_byte_identical": True, "artifacts": artifacts,
               "data_dir": str(args.history_output), "pod_dir": str(args.pod_output),
               "warning_for_future_training": "Do not reuse old flattened li/geometry target stores without explicit native-frame remapping. Do not enable native-count weighting: all training frame counts are now one-to-one."}
    write_json(args.results / "summary.json", summary)
    (args.results / "README.md").write_text(
        "# Native training data with frozen POD normalization\n\n"
        f"Prepared {summary['training_shots']:,} existing training shots at every native timestamp: "
        f"{summary['training_frames']:,} frames ({summary['density_multiplier']:.6f}× the 426,684-frame source).\n\n"
        f"Training data: `{args.history_output}`. Compact POD arrays: `{args.pod_output}`.\n\n"
        "All 426,684 previously cached training frames have bitwise identical raw inputs, history inputs, "
        "flux/scalar targets, native times and physical grids. Their compact input/coefficient/scalar rows "
        "are reused bitwise. Only the 1,069,723 added native frames are projected using float32 batches "
        "within each shot; the exact batch policy is recorded in summary.json and basis metadata. "
        "All 1,496,407 frame identities, times and target-content hashes are recorded per shot in projection_state.json.\n\n"
        "The input normalization, scalar/flux statistics and rank-128 POD basis are copied unchanged from "
        "`data/pod_history128_robust_v4`. No fit or model training was run. Validation and legacy arrays "
        "are reused byte-for-byte; audit entries are omitted and no audit labels were opened. "
        "The current training manifest hash changes to describe the denser view; the original basis metadata, "
        "basis-sample indices, source manifests and preparation source files are preserved in provenance/.\n\n"
        "Use the new data-dir and pod-dir together. Old li/geometry flattened target stores are incompatible "
        "until explicitly remapped by shot and native frame; native-count weighting is unnecessary. "
        "This cache intentionally omits large consolidated raw-X/psi duplicates: the POD DeepONet trainer "
        "reads compact arrays directly. A future full-psi consumer should stream or explicitly budget memory.\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "artifacts"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
