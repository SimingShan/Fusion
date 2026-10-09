#!/usr/bin/env python3
"""Validate a complete machine submission against pinned raw data and make its ZIP.

Only timestamp/grid columns are read from raw Parquet. Validation is independent
of the processed-data manifest and never accesses hidden test targets or a login.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
import zipfile

import numpy as np
import pyarrow.parquet as pq

CONFIG = "diii_d_public_test"
EXPECTED_SHOTS = 874
MACHINE_SHOTS = {"diii_d_public_test": 874, "mast_public_test": 1206}
REPOSITORY = "Sophelio/fusion-equilibrium-challenge"
SCALARS = ("q95", "betaN")
CHUNK_SIZE = 8 * 1024 * 1024


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(CHUNK_SIZE), b""):
            digest.update(block)
    return digest.hexdigest()


def array_hash(value):
    return hashlib.sha256(np.asarray(value, dtype="<f8").tobytes()).hexdigest()


def write_json_atomic(path, value):
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def validate_prediction(pred, raw_dir, expected_shots=EXPECTED_SHOTS, config=CONFIG):
    """Validate every array; expected_shots is parameterized for small local checks."""
    manifest_path = raw_dir / "download_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("repo") != REPOSITORY:
        raise ValueError(f"Unexpected dataset repository: {manifest.get('repo')!r}")
    revision = manifest.get("revision", "")
    if not re.fullmatch(r"[0-9a-fA-F]{40}", revision):
        raise ValueError("download_manifest.json must pin a full 40-character commit SHA")
    inventory = sorted((item for item in manifest["files"]
                        if item["path"].startswith(f"data/{config}/")
                        and item["path"].endswith(".parquet")), key=lambda item: item["path"])
    if len(inventory) != expected_shots:
        raise ValueError(f"Pinned manifest has {len(inventory)} test shots; expected {expected_shots}")
    paths = [item["path"] for item in inventory]
    if len(set(paths)) != len(paths):
        raise ValueError("Pinned manifest contains duplicate test paths")
    expected_keys = {f"shot_{index:04d}_{name}" for index in range(expected_shots)
                     for name in ("psirz", *SCALARS)}
    with zipfile.ZipFile(pred) as archive:
        names = archive.namelist()
        expected_members = {key + ".npy" for key in expected_keys}
        if len(names) != len(set(names)):
            raise ValueError("Prediction NPZ contains duplicate archive members")
        missing, extra = expected_members - set(names), set(names) - expected_members
        if missing or extra:
            raise ValueError(f"NPZ key mismatch: {len(missing)} missing, {len(extra)} extra; "
                             f"missing examples={sorted(missing)[:3]}, extra examples={sorted(extra)[:3]}")
    shots, dtype_counts, total_frames = [], {}, 0
    with np.load(pred, allow_pickle=False) as arrays:
        for index, item in enumerate(inventory):
            relative = Path(item["path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"Invalid dataset path: {relative}")
            path = raw_dir / relative
            if not path.is_file() or path.stat().st_size != item["size"]:
                raise ValueError(f"Missing or incomplete pinned raw shot: {path}")
            parquet = pq.ParquetFile(path)
            if parquet.metadata.num_rows != 1:
                raise ValueError(f"Expected one row per official shot: {path}")
            columns = ["efit_times", "efit_grid_R", "efit_grid_Z"]
            if not set(columns).issubset(parquet.schema_arrow.names):
                raise ValueError(f"Missing target timestamp/grid metadata: {path}")
            row = parquet.read(columns=columns).to_pylist()[0]
            times = np.asarray(row["efit_times"], dtype=np.float64)
            R, Z = (np.asarray(row[key], dtype=np.float64) for key in columns[1:])
            if times.ndim != 1 or not len(times) or not np.isfinite(times).all():
                raise ValueError(f"Invalid EFIT timestamps: {path}")
            if not np.all(np.diff(times) > 0):
                raise ValueError(f"EFIT timestamps must be strictly increasing: {path}")
            for name, grid in (("R", R), ("Z", Z)):
                if grid.shape != (65,) or not np.isfinite(grid).all() or not np.all(np.diff(grid) > 0):
                    raise ValueError(f"Invalid native {name} grid in {path}: {grid.shape}")
            n = len(times)
            for name in ("psirz", *SCALARS):
                key = f"shot_{index:04d}_{name}"
                value = arrays[key]
                shape = (n, 65, 65) if name == "psirz" else (n,)
                if value.shape != shape:
                    raise ValueError(f"{key}: shape {value.shape}, expected {shape} from {relative}")
                if not np.issubdtype(value.dtype, np.floating):
                    raise ValueError(f"{key}: expected floating dtype, received {value.dtype}")
                if not np.isfinite(value).all():
                    raise ValueError(f"{key}: contains NaN or infinity")
                dtype_counts[str(value.dtype)] = dtype_counts.get(str(value.dtype), 0) + 1
            shots.append({"prediction_index": index, "raw_path": item["path"], "frames": n,
                          "first_efit_time_ms": float(times[0]), "last_efit_time_ms": float(times[-1]),
                          "efit_times_sha256": array_hash(times), "grid_R_sha256": array_hash(R),
                          "grid_Z_sha256": array_hash(Z), "raw_expected_sha256": item.get("sha256")})
            total_frames += n
            if (index + 1) % 50 == 0 or index + 1 == expected_shots:
                print(f"Validated {index + 1}/{expected_shots} shots; {total_frames} frames", flush=True)
    return {"status": "passed", "config": config, "expected_shots": expected_shots,
            "validated_shots": len(shots), "expected_arrays": 3 * expected_shots,
            "validated_arrays": len(expected_keys), "frames": total_frames,
            "grid_shape": [65, 65], "grid_orientation": "rows=Z, columns=R",
            "all_arrays_floating": True, "all_arrays_finite": True, "extra_arrays": 0,
            "dtype_array_counts": dtype_counts, "dataset_repository": REPOSITORY,
            "dataset_revision": revision, "download_manifest_sha256": sha256_file(manifest_path),
            "raw_metadata_columns_read": ["efit_times", "efit_grid_R", "efit_grid_Z"],
            "raw_integrity_check": "Pinned manifest paths and file sizes; projected metadata. "
                                   "Full raw-file SHA256 checks are performed by the downloader, not repeated here.",
            "shot_order": "Lexicographically sorted official parquet paths from pinned download manifest",
            "order_validation_limit": "Prediction keys and frame counts are checked against this order. "
                                      "The official NPZ format contains no independent shot identities or timestamps "
                                      "with which to verify the semantic ordering of predicted values.",
            "validation_type": "local structural validation; no hidden labels or leaderboard score",
            "shots": shots}


def file_signature(path):
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns, stat.st_ino


def package(pred, output, signature, config=CONFIG):
    """Wrap already-compressed NPZ without recompression; hash the exact payload."""
    temp = output.with_name(output.name + f".{os.getpid()}.tmp")
    digest = hashlib.sha256()
    try:
        with zipfile.ZipFile(temp, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
            member = zipfile.ZipInfo(f"{config}.npz")
            member.compress_type = zipfile.ZIP_STORED
            member.external_attr = 0o100644 << 16
            with pred.open("rb") as source, archive.open(member, "w", force_zip64=True) as sink:
                for block in iter(lambda: source.read(CHUNK_SIZE), b""):
                    digest.update(block)
                    sink.write(block)
        if file_signature(pred) != signature:
            raise ValueError("Predictions changed during validation or packaging; run again after inference finishes")
        zip_sha256 = sha256_file(temp)
        os.replace(temp, output)
    finally:
        temp.unlink(missing_ok=True)
    return {"input_sha256": digest.hexdigest(), "zip_sha256": zip_sha256,
            "input_bytes": pred.stat().st_size, "zip_bytes": output.stat().st_size,
            "archive_member": f"{config}.npz", "archive_compression": "ZIP_STORED"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--config", choices=tuple(MACHINE_SHOTS), default=CONFIG)
    args = parser.parse_args()
    if args.pred.resolve() == args.output.resolve():
        parser.error("--output must differ from --pred")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report_path = args.output.parent / "validation.json"
    if args.pred.resolve() == report_path.resolve() or args.output.resolve() == report_path.resolve():
        parser.error("Input/output must not use the reserved report name validation.json")
    started = time.monotonic()
    report = {"status": "failed", "prediction_path": str(args.pred.resolve()),
              "submission_zip": str(args.output.resolve()),
              "raw_directory": str(args.raw_dir.resolve()),
              "created_utc": datetime.now(timezone.utc).isoformat()}
    try:
        signature = file_signature(args.pred)
        report.update(validate_prediction(args.pred, args.raw_dir, MACHINE_SHOTS[args.config], args.config))
        print("Writing direct-upload ZIP and computing SHA256 hashes...", flush=True)
        report.update(package(args.pred, args.output, signature, args.config))
        report["elapsed_seconds"] = time.monotonic() - started
        write_json_atomic(report_path, report)
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}",
                      elapsed_seconds=time.monotonic() - started)
        write_json_atomic(report_path, report)
        print(f"Validation/package failed: {exc}", file=sys.stderr)
        return 1
    print(f"Ready: {args.output} ({report['validated_shots']} shots, {report['frames']} frames); "
          f"report: {report_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
