#!/usr/bin/env python3
"""Independently summarize every native frame in a complete MAST export."""
from pathlib import Path
import argparse
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.challenge2.reuse_valid_export import digest, read, write, checked_plan
from scripts.challenge2.summarize_native_coverage import summarize


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise ValueError("Preserve existing coverage evidence")
    run = a.run.resolve()
    plan, report = checked_plan(run), read(run / "mast_export.json")
    validation = read(run / "structural_validation.json")
    if (report["status"] != "complete" or report["scope"] != "full"
            or report["plan_sha256"] != digest(run / "export_plan.json")
            or report["prediction_sha256"] != digest(run / "mast_public_test.npz")
            or validation["status"] != "passed" or validation["validated_shots"] != 1206
            or validation["frames"] != 69075):
        raise ValueError("Only a completely validated native export can be reviewed")
    records, receipts = [], []
    for index, item in enumerate(plan["shots"]):
        path = run / "shots" / f"{index:04d}.json"
        receipt = read(path)
        if (receipt["status"] != "complete" or receipt["index"] != index or receipt["raw_path"] != item["path"]
                or receipt["plan_sha256"] != report["plan_sha256"]
                or len(receipt["records"]) != receipt["frames"]):
            raise ValueError("Incomplete or inconsistent native sequence")
        receipts.append({"index": index, "sha256": digest(path)})
        records.extend(dict(row, shot=Path(item["path"]).stem, machine="MAST") for row in receipt["records"])
    if len(records) != 69075 or len({(x["shot"], x["frame"]) for x in records}) != 69075:
        raise ValueError("Native frame coverage changed")
    coverage = summarize(records, plan["max_seconds"])
    for key in ("shots", "frames", "converged", "scalar_valid", "runtime_limited_frames", "observed_pressure_retry_frames"):
        if coverage[key] != report[key]:
            raise ValueError("Export and independently counted coverage disagree")
    output = {"scope": "full", "complete_native_queries": True, "plan_sha256": report["plan_sha256"],
              "prediction_sha256": report["prediction_sha256"], "prior_sha256": report["prior_sha256"],
              "source_sha256": plan["source_sha256"], "structural_validation_sha256": digest(run / "structural_validation.json"),
              "machines": {"MAST": coverage}, "receipt_sha256": receipts, "mast_labels_read": False,
              "audit_labels_read": False, "mast_accuracy_established": False, "eligible_for_submission": False,
              "review_source_sha256": digest(__file__),
              "limitations": "Finite fields, derived scalars and numerical residuals do not establish hidden-target accuracy"}
    write(a.output, output)
    print(json.dumps({k: v for k, v in coverage.items() if k not in ("statistics", "statuses")}), flush=True)


if __name__ == "__main__":
    main()
