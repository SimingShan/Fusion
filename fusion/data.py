"""Reproducible, input-only features and restartable DIII-D shot caches.

Parquet nested arrays are converted from Arrow buffers, without materializing
the high-frequency traces as Python lists. All split decisions are shot-level.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from starter.data_fixes import fix_d3d_ip_times


CACHE_VERSION = 1
COILS = ["ECOILA", "bcoil"] + [f"F{i}{side}" for i in range(1, 10) for side in "AB"] + ["plasma_current"]
SCALAR_NAMES = ["q95", "betaN"]
TARGET_COLUMNS = ["efit_psirz", "efit_q95", "efit_beta_n"]
IDENTITY_COLUMNS = ["shot", "shot_id", "shot_number", "record_id", "id", "source"]
INPUT_COLUMNS = (
    ["efit_times", "efit_grid_R", "efit_grid_Z", "magnetics_time"]
    + [f"magnetics_{coil}" for coil in COILS]
    + [f"magnetics_{coil}_times" for coil in COILS]
    + [f"thomson_{system}_{field}" for system in ("core", "edge") for field in ("times", "Te", "ne")]
)


def feature_names() -> list[str]:
    names = [f"magnetics_{c}" for c in COILS] + [f"magnetics_{c}_observed" for c in COILS]
    for system, bins in (("core", 16), ("edge", 8)):
        for measure in ("Te", "ne"):
            prefix = f"thomson_{system}_{measure}"
            names.extend(f"{prefix}_log1p_bin_{j:02d}" for j in range(bins))
            names.extend(f"{prefix}_observed_bin_{j:02d}" for j in range(bins))
            names.extend([f"{prefix}_log1p_mean", f"{prefix}_log1p_max", f"{prefix}_coverage"])
    return names + ["time_seconds"]


def _arrow_array_to_numpy(array: Any) -> np.ndarray:
    """Flatten rectangular Arrow list arrays using offsets, then restore shape."""
    import pyarrow as pa

    if pa.types.is_list(array.type) or pa.types.is_large_list(array.type) or pa.types.is_fixed_size_list(array.type):
        if len(array) == 0:
            return np.empty((0,), dtype=np.float64)
        if pa.types.is_fixed_size_list(array.type):
            lengths = np.full(len(array), array.type.list_size)
        else:
            lengths = np.diff(array.offsets.to_numpy(zero_copy_only=False))
        if array.null_count == 0 and np.all(lengths == lengths[0]):
            child = _arrow_array_to_numpy(array.flatten())
            return child.reshape((len(array), int(lengths[0])) + child.shape[1:])
        # This path is only for small ragged or null diagnostic profiles.
        converted = [arrow_to_numpy(item) for item in array]
        shapes = [item.shape for item in converted if isinstance(item, np.ndarray)]
        if shapes and all(shape == shapes[0] for shape in shapes):
            return np.stack([np.full(shapes[0], np.nan) if item is None else item for item in converted])
        raise ValueError("Ragged nested Arrow array cannot be interpreted as a rectangular diagnostic")
    return array.to_numpy(zero_copy_only=False)


def arrow_to_numpy(value: Any) -> Any:
    """Convert one Arrow scalar; only textual/scalar metadata uses as_py()."""
    if not value.is_valid:
        return None
    if hasattr(value, "values"):
        return _arrow_array_to_numpy(value.values)
    return value.as_py()


def interpolate_signal(times: Any, values: Any, query: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Finite, sorted interpolation with endpoint values and an observed mask.

    The observed mask is zero outside the available time interval; unavailable
    channels have zero values and masks. Duplicate times retain the first value.
    """
    result = np.zeros(len(query), dtype=np.float32)
    observed = np.zeros(len(query), dtype=np.float32)
    if times is None or values is None:
        return result, observed
    times = np.asarray(times, dtype=np.float64).ravel()
    values = np.asarray(values, dtype=np.float64).ravel()
    if len(times) != len(values):
        raise ValueError(f"Signal time/value lengths differ: {len(times)} versus {len(values)}")
    finite = np.isfinite(times) & np.isfinite(values)
    if not finite.any():
        return result, observed
    times, values = times[finite], values[finite]
    if np.any(np.diff(times) <= 0):
        order = np.argsort(times, kind="stable")
        times, values = times[order], values[order]
        times, unique = np.unique(times, return_index=True)
        values = values[unique]
    result[:] = np.interp(query, times, values)
    observed[:] = (query >= times[0]) & (query <= times[-1])
    return result, observed


def _thomson_features(row: dict[str, Any], system: str, measure: str, bins: int, query: np.ndarray) -> list[np.ndarray]:
    profiles = row.get(f"thomson_{system}_{measure}")
    times = row.get(f"thomson_{system}_times")
    empty = [np.zeros((len(query), bins), dtype=np.float32)] * 2 + [np.zeros((len(query), 3), dtype=np.float32)]
    if profiles is None or times is None:
        return empty
    profiles = np.asarray(profiles, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64).ravel()
    if profiles.size == 0 or times.size == 0:
        return empty
    if profiles.ndim == 1:
        profiles = profiles[:, None]
    if profiles.ndim != 2:
        raise ValueError(f"Unexpected Thomson profile shape {profiles.shape}")
    if profiles.shape[0] != len(times) and profiles.shape[1] == len(times):
        profiles = profiles.T
    if profiles.shape[0] != len(times):
        raise ValueError(f"Thomson time/profile mismatch: {len(times)} and {profiles.shape}")
    # Negative temperature/density values are invalid measurements.
    profiles = np.where(profiles >= 0, profiles, np.nan)
    interpolated = [interpolate_signal(times, profiles[:, channel], query) for channel in range(profiles.shape[1])]
    values = np.column_stack([pair[0] for pair in interpolated])
    observed = np.column_stack([pair[1] for pair in interpolated])
    positions = np.linspace(0, profiles.shape[1] - 1, bins)
    lower = np.floor(positions).astype(int)
    upper = np.ceil(positions).astype(int)
    fraction = (positions - lower)[None, :]
    binned = values[:, lower] * (1 - fraction) + values[:, upper] * fraction
    bin_mask = observed[:, lower] * (1 - fraction) + observed[:, upper] * fraction
    # Statistics omit missing and out-of-coverage channels rather than imputing zero.
    count = observed.sum(axis=1)
    mean = (values * observed).sum(axis=1) / np.maximum(count, 1)
    maximum = np.where(observed > 0, values, 0).max(axis=1)
    stats = np.column_stack((np.log1p(mean), np.log1p(maximum), count / profiles.shape[1]))
    return [np.log1p(np.maximum(binned, 0)), bin_mask, stats]


def extract_features(row: dict[str, Any], frame_indices: np.ndarray | None = None) -> np.ndarray:
    """Extract only allowed coil currents, Thomson data, and query times."""
    query = np.asarray(row["efit_times"], dtype=np.float64).ravel()
    if frame_indices is not None:
        query = query[frame_indices]
    shared_time = row.get("magnetics_time")
    ip_times = row.get("magnetics_plasma_current_times", shared_time)
    ip_values = row.get("magnetics_plasma_current")
    if ip_times is not None and ip_values is not None and shared_time is not None:
        if len(ip_times) and len(ip_values) and np.isfinite(ip_values).any():
            # Official v1.1.0 erratum correction; depends on input columns only.
            ip_times = fix_d3d_ip_times({**row, "magnetics_plasma_current_times": ip_times})
    coil_values, coil_masks = [], []
    for coil in COILS:
        times = ip_times if coil == "plasma_current" else row.get(f"magnetics_{coil}_times")
        if times is None:
            times = shared_time
        value, mask = interpolate_signal(times, row.get(f"magnetics_{coil}"), query)
        coil_values.append(value)
        coil_masks.append(mask)
    pieces = [np.column_stack(coil_values), np.column_stack(coil_masks)]
    for system, bins in (("core", 16), ("edge", 8)):
        for measure in ("Te", "ne"):
            pieces.extend(_thomson_features(row, system, measure, bins, query))
    pieces.append(query[:, None] / 1000.0)
    X = np.concatenate(pieces, axis=1).astype(np.float32)
    if X.shape[1] != len(feature_names()) or not np.isfinite(X).all():
        raise ValueError(f"Invalid feature matrix: shape {X.shape}, expected {len(feature_names())} finite columns")
    return X


def read_manifest(processed_dir: str | Path) -> dict[str, Any]:
    with (Path(processed_dir) / "manifest.json").open() as handle:
        return json.load(handle)


def iter_cached_shots(processed_dir: str | Path, split: str) -> Iterator[tuple[dict[str, Any], dict[str, np.ndarray]]]:
    root = Path(processed_dir)
    for entry in read_manifest(root)[split]:
        with np.load(root / entry["path"], allow_pickle=False) as archive:
            yield entry, {key: archive[key] for key in archive.files}


def load_split(processed_dir: str | Path, split: str) -> dict[str, np.ndarray]:
    """Concatenate a cached split; metadata retains original shot/frame identity."""
    collected: dict[str, list[np.ndarray]] = {}
    for entry, shot in iter_cached_shots(processed_dir, split):
        for key in ("X", "psi", "scalars", "times", "frame_indices"):
            if key in shot:
                collected.setdefault(key, []).append(shot[key])
        collected.setdefault("shot_index", []).append(np.full(len(shot["X"]), entry["shot_index"], dtype=np.int64))
        if "R" not in collected:
            collected["R"], collected["Z"] = [shot["R"]], [shot["Z"]]
        elif not np.array_equal(collected["R"][0], shot["R"]) or not np.array_equal(collected["Z"][0], shot["Z"]):
            raise ValueError("load_split requires a common coordinate grid; use iter_cached_shots for variable grids")
    if not collected:
        raise ValueError(f"No prepared shots in split {split}")
    return {key: values[0] if key in ("R", "Z") else np.concatenate(values) for key, values in collected.items()}


def discover_shards(raw_dir: str | Path, config: str) -> list[Path]:
    """Find a contiguous prefix of completed shards, preserving official order."""
    raw_dir = Path(raw_dir)
    download_manifest = raw_dir / "download_manifest.json"
    if download_manifest.exists():
        inventory = json.loads(download_manifest.read_text())
        # The published repository uses one parquet file per shot. Canonical
        # indices come from the complete remote inventory, never arrival order.
        expected = sorted((item for item in inventory["files"] if item["path"].startswith(f"data/{config}/") and item["path"].endswith(".parquet")), key=lambda item: item["path"])
        return [raw_dir / item["path"] for item in expected if (raw_dir / item["path"]).exists() and (raw_dir / item["path"]).stat().st_size == item["size"]]
    candidates = sorted(path for path in raw_dir.rglob("*.parquet") if config in path.parts)
    contiguous = []
    for path in candidates:
        match = re.search(r"-(\d+)-of-(\d+)\.parquet$", path.name)
        if match is not None and int(match.group(1)) != len(contiguous):
            break
        contiguous.append(path)
    return contiguous


def parquet_inventory(paths: list[Path], raw_dir: str | Path) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    raw_dir = Path(raw_dir)
    download_manifest = raw_dir / "download_manifest.json"
    canonical = None
    if download_manifest.exists() and paths:
        config = paths[0].parent.name
        downloaded = json.loads(download_manifest.read_text())
        canonical = {name: index for index, name in enumerate(sorted(item["path"] for item in downloaded["files"] if item["path"].startswith(f"data/{config}/") and item["path"].endswith(".parquet")))}
    result, offset = [], 0
    for path in paths:
        metadata = pq.ParquetFile(path).metadata
        relative = str(path.relative_to(raw_dir))
        if canonical is not None:
            if metadata.num_rows != 1:
                raise ValueError(f"Published per-shot file has {metadata.num_rows} rows: {relative}")
            offset = canonical[relative]
        result.append({"path": relative, "n_rows": metadata.num_rows, "offset": offset})
        offset += metadata.num_rows
    return result


def _write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    os.replace(temporary, path)


def _cache_one(row: dict[str, Any], output: Path, split: str, index: int, row_index: int, shard: str, max_frames: int, targets: bool) -> dict[str, Any]:
    times = np.asarray(row["efit_times"], dtype=np.float64).ravel()
    if not len(times) or not np.isfinite(times).all():
        raise ValueError(f"Shot {index} has no finite EFIT time grid")
    frames = np.arange(len(times)) if max_frames <= 0 else np.unique(np.linspace(0, len(times) - 1, min(max_frames, len(times))).astype(np.int64))
    shot_id = next((str(row[key]) for key in IDENTITY_COLUMNS[:-1] if row.get(key) is not None), Path(shard).stem)
    relative = f"{split}/shot_{index:05d}.npz"
    path = output / relative
    arrays = {
        "X": extract_features(row, frames),
        "R": np.asarray(row["efit_grid_R"], dtype=np.float64).ravel(),
        "Z": np.asarray(row["efit_grid_Z"], dtype=np.float64).ravel(),
        "times": times[frames], "frame_indices": frames,
        "shot_index": np.asarray(index, dtype=np.int64), "shot_id": np.asarray(shot_id),
        "cache_version": np.asarray(CACHE_VERSION, dtype=np.int64),
    }
    if targets:
        psi = np.asarray(row["efit_psirz"], dtype=np.float32)
        if psi.shape != (len(times), len(arrays["Z"]), len(arrays["R"])):
            raise ValueError(f"Shot {index} flux/grid shape mismatch: {psi.shape}")
        arrays["psi"] = psi[frames]
        arrays["scalars"] = np.column_stack((row["efit_q95"], row["efit_beta_n"])).astype(np.float32)[frames]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".npz.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)
    return {"path": relative, "shot_index": index, "shot_id": shot_id, "n_frames": len(frames), "raw_frames": len(times), "shard": shard, "row": row_index}


def prepare_dataset(raw_dir: str | Path, output: str | Path, train_shots: int = 256, val_shots: int = 32, train_frames: int = 64, val_frames: int = 32, seed: int = 2026, train_only: bool = False, test_only: bool = False) -> dict[str, Any]:
    """Prepare selected training shots and every available public-test shot.

    Selection uses the first train_shots + val_shots official rows, followed by a
    seeded shot-level shuffle. Reusing an output directory requires identical
    preparation settings. Existing complete shots are reused after interruption.
    """
    import pyarrow.parquet as pq

    raw_dir, output = Path(raw_dir).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    settings = {"train_shots": train_shots, "val_shots": val_shots, "train_frames": train_frames, "val_frames": val_frames, "seed": seed}
    manifest_path = output / "manifest.json"
    manifest = read_manifest(output) if manifest_path.exists() else {"version": CACHE_VERSION, "settings": settings, "feature_names": feature_names(), "scalar_names": SCALAR_NAMES, "train": [], "val": [], "test": [], "inventory": {}}
    if manifest["version"] != CACHE_VERSION or manifest["settings"] != settings:
        raise ValueError("Cache version/settings differ; use a new output directory to preserve reproducible splits")
    download_manifest_path = raw_dir / "download_manifest.json"
    if download_manifest_path.exists():
        download_manifest = json.loads(download_manifest_path.read_text())
        dataset = {key: download_manifest[key] for key in ("repo", "revision") if key in download_manifest}
        if manifest.get("dataset", dataset) != dataset:
            raise ValueError("Downloaded dataset revision changed; use a new output directory")
        manifest["dataset"] = dataset
        manifest["expected_shots"] = {config: sum(item["path"].startswith(f"data/{config}/") and item["path"].endswith(".parquet") for item in download_manifest["files"]) for config in ("diii_d_train", "diii_d_public_test")}
    configs = []
    if not test_only:
        configs.append(("diii_d_train", True))
    if not train_only:
        configs.append(("diii_d_public_test", False))
    for config, targets in configs:
        inventory = parquet_inventory(discover_shards(raw_dir, config), raw_dir)
        if not inventory:
            print(f"No completed contiguous shards for {config}; skipping", flush=True)
            continue
        manifest["inventory"][config] = inventory
        available_indices = {shard["offset"] + j for shard in inventory for j in range(shard["n_rows"])}
        available = len(available_indices)
        if targets:
            total = train_shots + val_shots if train_shots > 0 else available
            if train_shots == 0 and "selection" in manifest:
                # Freeze an all-available selection across incremental downloads.
                total = len(manifest["selection"]["train"]) + len(manifest["selection"]["val"])
            if set(range(total)) - available_indices:
                missing = sorted(set(range(total)) - available_indices)
                raise ValueError(f"Need first {total} training rows; {len(missing)} missing (first missing indices: {missing[:10]})")
            if total <= val_shots:
                raise ValueError("At least one training shot must remain after validation selection")
            permutation = np.random.default_rng(seed).permutation(total)
            validation = set(permutation[:val_shots].tolist())
            assignments = {i: "val" if i in validation else "train" for i in range(total)}
            manifest["selection"] = {"method": "first_official_rows_then_seeded_shot_holdout", "train": [i for i in range(total) if i not in validation], "val": sorted(validation)}
        else:
            assignments = {i: "test" for i in sorted(available_indices)}
        known = {entry["shot_index"]: entry for split in (("train", "val") if targets else ("test",)) for entry in manifest[split]}
        _write_json_atomic(manifest_path, manifest)
        for shard in inventory:
            needed = {i for i in assignments if shard["offset"] <= i < shard["offset"] + shard["n_rows"] and (i not in known or not (output / known[i]["path"]).exists())}
            if not needed:
                continue
            parquet = pq.ParquetFile(raw_dir / shard["path"])
            available_columns = set(parquet.schema_arrow.names)
            columns = [column for column in IDENTITY_COLUMNS + INPUT_COLUMNS + (TARGET_COLUMNS if targets else []) if column in available_columns]
            required = {"efit_times", "efit_grid_R", "efit_grid_Z"} | (set(TARGET_COLUMNS) if targets else set())
            if required - available_columns:
                raise ValueError(f"Required columns absent in {shard['path']}: {required - available_columns}")
            print(f"Preparing {config}: {len(needed)} shots from {shard['path']}", flush=True)
            # One shot per batch keeps large raw current traces bounded in memory.
            for row_index, batch in enumerate(parquet.iter_batches(batch_size=1, columns=columns, use_threads=False)):
                index = shard["offset"] + row_index
                if index not in needed:
                    if index > max(needed):
                        break
                    continue
                row = {column: arrow_to_numpy(batch.column(j)[0]) for j, column in enumerate(columns)}
                split = assignments[index]
                max_frames = (train_frames if split == "train" else val_frames) if targets else 0
                entry = _cache_one(row, output, split, index, row_index, shard["path"], max_frames, targets)
                manifest[split] = [item for item in manifest[split] if item["shot_index"] != index] + [entry]
                manifest[split].sort(key=lambda item: item["shot_index"])
                known[index] = entry
                _write_json_atomic(manifest_path, manifest)
                del row, batch
                if (len(manifest["train"]) + len(manifest["val"]) if targets else len(manifest["test"])) % 10 == 0:
                    print(f"Cached train={len(manifest['train'])}, val={len(manifest['val'])}, test={len(manifest['test'])}", flush=True)
    return manifest
