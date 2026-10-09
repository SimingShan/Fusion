"""Input-only DIII-D features and scalable, reproducible shot-level caches.

The first 151 columns exactly reproduce the original baseline. Extra columns
preserve every Thomson channel and its physical location, coil geometry, and
nearby current measurements. No label-derived signal is read into features.
"""
from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any

import numpy as np

from fusion.data import (COILS, IDENTITY_COLUMNS, INPUT_COLUMNS, SCALAR_NAMES,
                         TARGET_COLUMNS, arrow_to_numpy, extract_features,
                         feature_names as baseline_feature_names,
                         interpolate_signal, read_manifest)
from starter.data_fixes import fix_d3d_ip_times


VERSION = 1
THOMSON_CAPACITY = {"core": 64, "edge": 32}
CURRENT_OFFSETS_MS = (-100, -20, 20, 100)
GEOMETRY_FIELDS = ("R", "Z", "width", "height", "angle1", "angle2")
GEOMETRY_COLUMNS = (["thomson_core_R", "thomson_edge_spatial", "thomson_chord_name",
                     "thomson_chord_R", "thomson_chord_Z", "coil_input_column"]
                    + [f"coil_{field}" for field in GEOMETRY_FIELDS])
SPLITS = ("train", "val", "audit", "legacy_val", "test")


def feature_names() -> list[str]:
    names = baseline_feature_names()
    for system, capacity in THOMSON_CAPACITY.items():
        for measure in ("Te", "ne"):
            for field in ("log1p", "observed"):
                names += [f"thomson_{system}_{measure}_{field}_channel_{i:02d}" for i in range(capacity)]
        for field in ("R", "Z", "geometry_observed"):
            names += [f"thomson_{system}_{field}_channel_{i:02d}" for i in range(capacity)]
    for field in (*GEOMETRY_FIELDS, "geometry_observed"):
        names += [f"coil_{coil}_{field}" for coil in COILS]
    for offset in CURRENT_OFFSETS_MS:
        for field in ("value", "observed"):
            names += [f"magnetics_{coil}_{field}_offset_{offset:+d}ms" for coil in COILS]
    return names


def _profiles(row: dict[str, Any], system: str, measure: str) -> tuple[np.ndarray, np.ndarray]:
    values, times = row.get(f"thomson_{system}_{measure}"), row.get(f"thomson_{system}_times")
    if values is None or times is None:
        return np.empty((0, 0)), np.empty(0)
    values, times = np.asarray(values, dtype=np.float64), np.asarray(times, dtype=np.float64).ravel()
    if not values.size or not times.size:
        return np.empty((0, 0)), np.empty(0)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2:
        raise ValueError(f"Unexpected {system}/{measure} profile shape {values.shape}")
    if values.shape[0] != len(times) and values.shape[1] == len(times):
        values = values.T
    if values.shape[0] != len(times):
        raise ValueError(f"Thomson {system}/{measure} time/profile mismatch")
    if values.shape[1] > THOMSON_CAPACITY[system]:
        raise ValueError(f"{system} has {values.shape[1]} channels: increase capacity in a new cache version")
    return np.where(values >= 0, values, np.nan), times


def _chord_geometry(row: dict[str, Any], system: str, capacity: int) -> np.ndarray:
    """Align physical coordinates by the channel suffix, never by mixed-group order."""
    geometry = np.zeros((3, capacity), dtype=np.float32)
    group = "core" if system == "core" else "tangential"
    names = row.get("thomson_chord_name")
    radii, heights = row.get("thomson_chord_R"), row.get("thomson_chord_Z")
    if names is None or radii is None or heights is None:
        return geometry
    if not (len(names) == len(radii) == len(heights)):
        raise ValueError("Thomson chord geometry lengths disagree")
    seen = set()
    for name, radius, height in zip(names, radii, heights):
        match = re.fullmatch(rf"TS_{group}_.*_(\d+)", str(name))
        if match is None:
            continue
        channel = int(match.group(1))
        if channel in seen or channel >= capacity:
            raise ValueError(f"Duplicate/out-of-capacity {group} chord index {channel}")
        seen.add(channel)
        if np.isfinite(radius) and np.isfinite(height):
            geometry[:, channel] = (radius, height, 1)
    return geometry


def extract_advanced_features(row: dict[str, Any], frame_indices: np.ndarray | None = None) -> np.ndarray:
    query = np.asarray(row["efit_times"], dtype=np.float64).ravel()
    if frame_indices is not None:
        query = query[frame_indices]
    pieces = [extract_features(row, frame_indices)]
    for system, capacity in THOMSON_CAPACITY.items():
        for measure in ("Te", "ne"):
            profiles, times = _profiles(row, system, measure)
            values = np.zeros((len(query), capacity), dtype=np.float32)
            observed = np.zeros_like(values)
            for channel in range(profiles.shape[1]):
                values[:, channel], observed[:, channel] = interpolate_signal(times, profiles[:, channel], query)
            pieces.extend((np.log1p(np.maximum(values, 0)), observed))
        geometry = _chord_geometry(row, system, capacity)
        pieces += [np.broadcast_to(values, (len(query), capacity)) for values in geometry]
    geometry = np.zeros((len(GEOMETRY_FIELDS) + 1, len(COILS)), dtype=np.float32)
    columns = row.get("coil_input_column")
    if columns is not None:
        column_index = {str(name): i for i, name in enumerate(columns)}
        for j, coil in enumerate(COILS):
            i = column_index.get(f"magnetics_{coil}")
            if i is None:
                continue
            values = [row.get(f"coil_{field}") for field in GEOMETRY_FIELDS]
            if all(value is not None and len(value) == len(columns) for value in values):
                coordinates = np.asarray([value[i] for value in values], dtype=np.float64)
                if np.isfinite(coordinates).all():
                    geometry[:, j] = [*coordinates, 1]
    pieces += [np.broadcast_to(values, (len(query), len(COILS))) for values in geometry]
    shared_time = row.get("magnetics_time")
    ip_times = row.get("magnetics_plasma_current_times", shared_time)
    ip_values = row.get("magnetics_plasma_current")
    if ip_times is not None and ip_values is not None and shared_time is not None:
        if len(ip_times) and len(ip_values) and np.isfinite(ip_values).any():
            ip_times = fix_d3d_ip_times({**row, "magnetics_plasma_current_times": ip_times})
    # Offline inference permits observations both before and after query time.
    for offset in CURRENT_OFFSETS_MS:
        values, masks = [], []
        for coil in COILS:
            times = ip_times if coil == "plasma_current" else row.get(f"magnetics_{coil}_times")
            if times is None:
                times = shared_time
            value, mask = interpolate_signal(times, row.get(f"magnetics_{coil}"), query + offset)
            values.append(value)
            masks.append(mask)
        pieces.extend((np.column_stack(values), np.column_stack(masks)))
    result = np.concatenate(pieces, axis=1).astype(np.float32)
    if result.shape != (len(query), len(feature_names())) or not np.isfinite(result).all():
        raise ValueError(f"Invalid advanced feature matrix {result.shape}")
    return result


def _json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def create_plan(raw_dir: str | Path, legacy_dir: str | Path, seed: int = 20260913,
                val_shots: int = 128, audit_shots: int = 128) -> dict[str, Any]:
    raw_dir, legacy_dir = Path(raw_dir), Path(legacy_dir)
    downloaded = json.loads((raw_dir / "download_manifest.json").read_text())
    inventory = {}
    for config in ("diii_d_train", "diii_d_public_test"):
        files = sorted((item for item in downloaded["files"]
                        if item["path"].startswith(f"data/{config}/") and item["path"].endswith(".parquet")),
                       key=lambda item: item["path"])
        inventory[config] = [{"path": item["path"], "offset": i, "n_rows": 1, "size": item["size"]}
                             for i, item in enumerate(files)]
    legacy = read_manifest(legacy_dir)
    legacy_val = {int(item["shot_index"]) for item in legacy["val"]}
    original = legacy_val | {int(item["shot_index"]) for item in legacy["train"]}
    all_indices = set(range(len(inventory["diii_d_train"])))
    if not original <= all_indices or not legacy_val:
        raise ValueError("Original split IDs are missing from canonical raw inventory")
    # New holdouts have never appeared in baseline training or model selection.
    candidates = np.asarray(sorted(all_indices - original), dtype=np.int64)
    if val_shots < 1 or audit_shots < 1 or val_shots + audit_shots >= len(candidates):
        raise ValueError("Need positive held-out splits and remaining new training shots")
    candidates = np.random.default_rng(seed).permutation(candidates)
    val, audit = set(candidates[:val_shots].tolist()), set(candidates[val_shots:val_shots + audit_shots].tolist())
    selection = {"train": sorted(all_indices - legacy_val - val - audit), "val": sorted(val),
                 "audit": sorted(audit), "legacy_val": sorted(legacy_val),
                 "test": list(range(len(inventory["diii_d_public_test"]))) }
    groups = [set(selection[key]) for key in SPLITS if key != "test"]
    if set.union(*groups) != all_indices or sum(map(len, groups)) != len(all_indices):
        raise AssertionError("Shot-level split overlap or omissions")
    return {"selection": selection, "inventory": inventory,
            "dataset": {key: downloaded[key] for key in ("repo", "revision") if key in downloaded},
            "legacy_manifest_sha256": hashlib.sha256((legacy_dir / "manifest.json").read_bytes()).hexdigest(),
            "previously_used_indices": sorted(original),
            "split_method": "preserve_legacy_val; seeded_new_val_and_audit_exclude_all_original_baseline_shots"}


def _cache_task(task: dict[str, Any]) -> dict[str, Any]:
    import pyarrow.parquet as pq

    output = Path(task["output"])
    relative = f"{task['split']}/shot_{task['index']:05d}.npz"
    path = output / relative
    meta_path = path.with_suffix(".json")
    if path.exists() and meta_path.exists():
        metadata = json.loads(meta_path.read_text())
        if metadata.get("settings_hash") == task["settings_hash"] and metadata.get("file_size") == path.stat().st_size:
            return metadata
    raw_path = Path(task["raw_dir"]) / task["shard"]
    if not raw_path.exists() or raw_path.stat().st_size != task["raw_size"]:
        raise ValueError(f"Incomplete raw file {raw_path}")
    parquet = pq.ParquetFile(raw_path)
    if parquet.metadata.num_rows != 1:
        raise ValueError(f"Expected per-shot Parquet, got {parquet.metadata.num_rows} rows")
    available = set(parquet.schema_arrow.names)
    targets = task["split"] != "test"
    required = {"efit_times", "efit_grid_R", "efit_grid_Z"} | (set(TARGET_COLUMNS) if targets else set())
    if required - available:
        raise ValueError(f"Missing columns: {required - available}")
    columns = list(dict.fromkeys(column for column in [*IDENTITY_COLUMNS, *INPUT_COLUMNS, *GEOMETRY_COLUMNS,
                                                       *(TARGET_COLUMNS if targets else [])] if column in available))
    batch = next(parquet.iter_batches(batch_size=1, columns=columns, use_threads=False))
    row = {column: arrow_to_numpy(batch.column(j)[0]) for j, column in enumerate(columns)}
    times = np.asarray(row["efit_times"], dtype=np.float64).ravel()
    if not len(times) or not np.isfinite(times).all():
        raise ValueError("Invalid native EFIT timestamps")
    maximum = task["max_frames"]
    frames = (np.arange(len(times), dtype=np.int64) if maximum == 0 else
              np.unique(np.linspace(0, len(times) - 1, min(maximum, len(times))).astype(np.int64)))
    if task.get("legacy_cache"):
        with np.load(task["legacy_cache"], allow_pickle=False) as old:
            frames = old["frame_indices"]
            if not np.array_equal(times[frames], old["times"]):
                raise ValueError("Legacy shot timestamps do not match raw shot identity")
    shot_id = next((str(row[key]) for key in IDENTITY_COLUMNS[:-1] if row.get(key) is not None), Path(task["shard"]).stem)
    arrays = {"X": extract_advanced_features(row, frames),
              "R": np.asarray(row["efit_grid_R"], dtype=np.float64).ravel(),
              "Z": np.asarray(row["efit_grid_Z"], dtype=np.float64).ravel(),
              "times": times[frames], "frame_indices": frames,
              "shot_index": np.asarray(task["index"], dtype=np.int64), "shot_id": np.asarray(shot_id),
              "cache_version": np.asarray(VERSION, dtype=np.int64)}
    if targets:
        psi = np.asarray(row["efit_psirz"], dtype=np.float32)
        if psi.shape != (len(times), len(arrays["Z"]), len(arrays["R"])):
            raise ValueError(f"Flux/grid shape mismatch {psi.shape}")
        arrays["psi"] = psi[frames]
        arrays["scalars"] = np.column_stack((row["efit_q95"], row["efit_beta_n"])).astype(np.float32)[frames]
        if not np.isfinite(arrays["psi"]).all():
            raise ValueError("Nonfinite flux target")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        # Stored ZIP members avoid repeatedly compressing smooth 65x65 targets.
        np.savez(handle, **arrays)
    os.replace(temporary, path)
    metadata = {"path": relative, "shot_index": task["index"], "shot_id": shot_id,
                "n_frames": len(frames), "raw_frames": len(times), "shard": task["shard"], "row": 0,
                "settings_hash": task["settings_hash"], "file_size": path.stat().st_size}
    _json_atomic(meta_path, metadata)
    return metadata


def consolidate_split(output: str | Path, split: str, allow_partial: bool = False) -> dict[str, Any]:
    """Write NumPy-format memmaps in canonical shot order using bounded memory."""
    output = Path(output)
    manifest = read_manifest(output)
    entries = manifest[split]
    if not entries:
        raise ValueError(f"Empty split {split}")
    if not allow_partial and len(entries) != len(manifest["selection"][split]):
        raise ValueError(f"Refusing to consolidate incomplete {split} split")
    fingerprint = _digest({"settings_hash": manifest["settings_hash"], "entries": entries})
    root = output / "consolidated"
    root.mkdir(exist_ok=True)
    meta_path = root / f"{split}.json"
    if meta_path.exists():
        metadata = json.loads(meta_path.read_text())
        if metadata.get("fingerprint") == fingerprint and all((output / path).exists() for path in metadata["paths"].values()):
            return metadata
    n = sum(entry["n_frames"] for entry in entries)
    with np.load(output / entries[0]["path"], allow_pickle=False) as first:
        keys = [key for key in ("X", "psi", "scalars", "times", "frame_indices") if key in first]
        specifications = {key: {"shape": [n, *first[key].shape[1:]], "dtype": str(first[key].dtype)} for key in keys}
        reference_R, reference_Z = first["R"], first["Z"]
    specifications["shot_index"] = {"shape": [n], "dtype": "int64"}
    maps = {}
    try:
        for key, spec in specifications.items():
            maps[key] = np.lib.format.open_memmap(root / f"{split}_{key}.partial.npy", mode="w+", dtype=spec["dtype"], shape=tuple(spec["shape"]))
        offsets = [0]
        for i, entry in enumerate(entries):
            start, end = offsets[-1], offsets[-1] + entry["n_frames"]
            with np.load(output / entry["path"], allow_pickle=False) as shot:
                if not (np.array_equal(shot["R"], reference_R) and np.array_equal(shot["Z"], reference_Z)):
                    raise ValueError("Consolidation requires the common DIII-D physical grid")
                for key in keys:
                    maps[key][start:end] = shot[key]
            maps["shot_index"][start:end] = entry["shot_index"]
            offsets.append(end)
            if (i + 1) % 500 == 0:
                print(f"Consolidated {split}: {i + 1}/{len(entries)} shots", flush=True)
        for value in maps.values():
            value.flush()
        maps.clear()
        paths = {}
        for key in specifications:
            relative = f"consolidated/{split}_{key}.npy"
            os.replace(root / f"{split}_{key}.partial.npy", output / relative)
            paths[key] = relative
        for key, value in (("offsets", np.asarray(offsets, dtype=np.int64)), ("R", reference_R), ("Z", reference_Z)):
            relative = f"consolidated/{split}_{key}.npy"
            temporary = root / f"{split}_{key}.partial.npy"
            with temporary.open("wb") as handle:
                np.save(handle, value)
            os.replace(temporary, output / relative)
            paths[key] = relative
        metadata = {"fingerprint": fingerprint, "n_frames": n, "n_shots": len(entries),
                    "complete_split": len(entries) == len(manifest["selection"][split]),
                    "paths": paths, "arrays": specifications}
        _json_atomic(meta_path, metadata)
        return metadata
    finally:
        maps.clear()


def load_consolidated(output: str | Path, split: str, mmap_mode: str = "r") -> dict[str, np.ndarray]:
    output = Path(output)
    metadata = json.loads((output / "consolidated" / f"{split}.json").read_text())
    return {key: np.load(output / relative, mmap_mode=mmap_mode, allow_pickle=False)
            for key, relative in metadata["paths"].items()}


def prepare_advanced_dataset(raw_dir: str | Path = "data/raw", output: str | Path = "data/advanced_v1",
                             legacy_dir: str | Path = "data/processed", seed: int = 20260913,
                             val_shots: int = 128, audit_shots: int = 128, train_frames: int = 64,
                             val_frames: int = 32, workers: int = 4, limit_shots: int = 0,
                             consolidate: bool = True, include_test: bool = False,
                             splits: list[str] | None = None, plan_only: bool = False) -> dict[str, Any]:
    raw_dir, output, legacy_dir = Path(raw_dir).resolve(), Path(output).resolve(), Path(legacy_dir).resolve()
    if train_frames < 0 or val_frames < 0 or workers < 1 or limit_shots < 0:
        raise ValueError("Frame caps/shot limit must be nonnegative and workers positive")
    plan = create_plan(raw_dir, legacy_dir, seed, val_shots, audit_shots)
    settings = {"seed": seed, "val_shots": val_shots, "audit_shots": audit_shots,
                "train_frames": train_frames, "val_frames": val_frames, "version": VERSION,
                "legacy_manifest_sha256": plan["legacy_manifest_sha256"],
                "source_sha256": {str(path.relative_to(Path(__file__).resolve().parents[1])):
                                   hashlib.sha256(path.read_bytes()).hexdigest()
                                   for path in (Path(__file__).resolve(), Path(__file__).resolve().with_name("data.py"),
                                                Path(__file__).resolve().parents[1] / "starter/data_fixes.py")}}
    settings_hash = _digest(settings)
    manifest_path = output / "manifest.json"
    manifest = {"version": VERSION, "settings": settings, "settings_hash": settings_hash,
                "feature_names": feature_names(), "baseline_feature_count": len(baseline_feature_names()),
                "scalar_names": SCALAR_NAMES, **plan,
                "expected_shots": {config: len(values) for config, values in plan["inventory"].items()},
                "consolidated": {}, **{split: [] for split in SPLITS}}
    if manifest_path.exists():
        previous = read_manifest(output)
        if previous["settings_hash"] != settings_hash or previous["dataset"] != plan["dataset"]:
            raise ValueError("Cache settings/source/dataset differ; use a new output directory")
        manifest = previous
    selected_splits = splits if splits is not None else ["legacy_val", "val", "train", "audit"] + (["test"] if include_test else [])
    if any(split not in SPLITS for split in selected_splits):
        raise ValueError(f"Unknown split: {selected_splits}")
    if plan_only:
        return manifest
    output.mkdir(parents=True, exist_ok=True)
    _json_atomic(manifest_path, manifest)
    legacy_lookup = {int(entry["shot_index"]): str(legacy_dir / entry["path"]) for entry in read_manifest(legacy_dir)["val"]}
    tasks = []
    for split in selected_splits:
        config = "diii_d_public_test" if split == "test" else "diii_d_train"
        known = {entry["shot_index"]: entry for entry in manifest[split]}
        for index in manifest["selection"][split]:
            if index in known:
                cached = output / known[index]["path"]
                if cached.exists() and cached.stat().st_size == known[index].get("file_size"):
                    continue
            shard = manifest["inventory"][config][index]
            tasks.append({"raw_dir": str(raw_dir), "output": str(output), "split": split, "index": index,
                          "shard": shard["path"], "raw_size": shard["size"], "settings_hash": settings_hash,
                          "max_frames": 0 if split == "test" else train_frames if split == "train" else val_frames,
                          "legacy_cache": legacy_lookup.get(index) if split == "legacy_val" else None})
    if limit_shots:
        tasks = tasks[:limit_shots]
    counts = ", ".join(f"{key}={len(manifest['selection'][key])}" for key in SPLITS)
    print(f"Plan: {counts}; "
          f"features={len(feature_names())}; pending_this_run={len(tasks)}", flush=True)
    completed = 0
    def accept(task: dict[str, Any], entry: dict[str, Any]) -> None:
        nonlocal completed
        split = task["split"]
        manifest[split] = [item for item in manifest[split] if item["shot_index"] != entry["shot_index"]] + [entry]
        manifest[split].sort(key=lambda item: item["shot_index"])
        completed += 1
        if completed % 32 == 0:
            _json_atomic(manifest_path, manifest)
            print(f"Prepared {completed}/{len(tasks)} new shots: " + ", ".join(f"{key}={len(manifest[key])}" for key in selected_splits), flush=True)
    try:
        if workers == 1:
            for task in tasks:
                accept(task, _cache_task(task))
        else:
            iterator = iter(tasks)
            with ProcessPoolExecutor(max_workers=workers) as executor:
                pending = {}
                for _ in range(min(len(tasks), workers * 2)):
                    task = next(iterator)
                    pending[executor.submit(_cache_task, task)] = task
                while pending:
                    done, _ = wait(pending, return_when=FIRST_COMPLETED)
                    for future in done:
                        accept(pending.pop(future), future.result())
                        task = next(iterator, None)
                        if task is not None:
                            pending[executor.submit(_cache_task, task)] = task
    finally:
        manifest["complete_splits"] = {split: len(manifest[split]) == len(manifest["selection"][split]) for split in SPLITS}
        _json_atomic(manifest_path, manifest)
    if consolidate:
        for split in selected_splits:
            if manifest["complete_splits"][split] and manifest[split]:
                manifest["consolidated"][split] = consolidate_split(output, split)
                _json_atomic(manifest_path, manifest)
    return manifest
