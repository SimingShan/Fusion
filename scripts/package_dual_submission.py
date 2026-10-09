#!/usr/bin/env python3
"""Validate and package both official machines in one direct-upload ZIP.

Only public timestamp/grid metadata is accessed. This command never submits.
"""
from datetime import datetime, timezone
import argparse
import hashlib
import os
from pathlib import Path
import sys
import time
import zipfile

from package_submission import (
    CHUNK_SIZE, MACHINE_SHOTS, file_signature, sha256_file,
    validate_prediction, write_json_atomic,
)


def package_pair(predictions, output, signatures):
    if set(predictions) != set(MACHINE_SHOTS):
        raise ValueError("Both official machine configurations are required")
    temp = output.with_name(output.name + f".{os.getpid()}.tmp")
    payloads = {}
    try:
        with zipfile.ZipFile(temp, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
            for config, pred in predictions.items():
                if file_signature(pred) != signatures[config]:
                    raise ValueError(f"{config} changed after validation")
                digest = hashlib.sha256()
                member = zipfile.ZipInfo(f"{config}.npz")
                member.compress_type = zipfile.ZIP_STORED
                member.external_attr = 0o100644 << 16
                with pred.open("rb") as src, archive.open(member, "w", force_zip64=True) as dst:
                    for block in iter(lambda: src.read(CHUNK_SIZE), b""):
                        digest.update(block)
                        dst.write(block)
                payloads[config] = {"path": str(pred.resolve()), "sha256": digest.hexdigest(),
                                    "bytes": pred.stat().st_size, "archive_member": member.filename}
        for config, pred in predictions.items():
            if file_signature(pred) != signatures[config]:
                raise ValueError(f"{config} changed during validation or packaging")
        checksum = sha256_file(temp)
        os.replace(temp, output)
    finally:
        temp.unlink(missing_ok=True)
    return {"payloads": payloads, "zip_sha256": checksum,
            "zip_bytes": output.stat().st_size, "archive_compression": "ZIP_STORED"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--diii-d", type=Path, required=True)
    p.add_argument("--mast", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    args = p.parse_args()
    predictions = {"diii_d_public_test": args.diii_d.resolve(), "mast_public_test": args.mast.resolve()}
    output = args.output.resolve()
    report_path = output.parent / "validation.json"
    paths = [*predictions.values(), output, report_path]
    if len(set(paths)) != len(paths):
        p.error("Input, output and reserved validation.json paths must all differ")
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    report = {"status": "failed", "submission_zip": str(output),
              "created_utc": datetime.now(timezone.utc).isoformat(),
              "validation_type": "dual-machine structural validation; no hidden labels or official score",
              "raw_directory": str(args.raw_dir.resolve()), "machines": {}}
    try:
        signatures = {k: file_signature(v) for k, v in predictions.items()}
        for config, pred in predictions.items():
            report["machines"][config] = validate_prediction(pred, args.raw_dir, MACHINE_SHOTS[config], config)
        report.update(package_pair(predictions, output, signatures))
        report.update(status="passed", elapsed_seconds=time.monotonic() - started)
        write_json_atomic(report_path, report)
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}",
                      elapsed_seconds=time.monotonic() - started)
        write_json_atomic(report_path, report)
        print(f"Dual-machine packaging failed: {exc}", file=sys.stderr)
        return 1
    print(f"Ready: {output}; {report['zip_bytes']:,} bytes; SHA256 {report['zip_sha256']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
