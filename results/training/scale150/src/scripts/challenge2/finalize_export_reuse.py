#!/usr/bin/env python3
"""Bind complete-export evidence to every explicitly reused sequence."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.challenge2.reuse_valid_export import digest, read, write, checked_plan


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    a = p.parse_args()
    run = a.run.resolve()
    plan, proof, report = checked_plan(run), read(run / "reuse_proof.json"), read(run / "mast_export.json")
    proof_hash = digest(run / "reuse_proof.json")
    parent = Path(proof["parent"])
    if (report["status"] != "complete" or report["scope"] != "full" or report["frames"] != 69075
            or report["scalar_valid"] != 69075 or report["shots"] != 1206 or report["runtime_limited_frames"]
            or report["plan_sha256"] != proof["new_plan_sha256"]
            or digest(run / "export_plan.json") != proof["new_plan_sha256"]
            or digest(parent / "export_plan.json") != proof["parent_plan_sha256"]
            or digest(proof["runner"]) != proof["runner_sha256"]
            or digest(run / "mast_public_test.npz") != report["prediction_sha256"]):
        raise ValueError("A complete, unchanged native export is required")
    for item in proof["evidence"]:
        if digest(item["path"]) != item["sha256"]:
            raise ValueError("Reuse evidence changed")
    reused, computed, retries, frames = [], [], 0, 0
    for index, item in enumerate(plan["shots"]):
        path = run / "shots" / f"{index:04d}.json"
        record = read(path)
        if (record["status"] != "complete" or record["index"] != index or record["raw_path"] != item["path"]
                or record["plan_sha256"] != proof["new_plan_sha256"] or record["frames"] != record["scalar_valid"]
                or record["runtime_limited_frames"] or digest(path.with_suffix(".npz")) != record["prediction_sha256"]):
            raise ValueError("Incomplete or changed native sequence")
        frames += record["frames"]
        retries += record["observed_pressure_retry_frames"]
        if "reuse" in record:
            reuse = record["reuse"]
            original = parent / "shots" / path.name
            before = read(original)
            if (reuse["proof_sha256"] != proof_hash or reuse["parent_plan_sha256"] != proof["parent_plan_sha256"]
                    or Path(reuse["parent_receipt"]) != original
                    or digest(original) != reuse["parent_receipt_sha256"]
                    or reuse["prediction_recomputed"] is not False or record["observed_pressure_retry_frames"]
                    or before["prediction_sha256"] != record["prediction_sha256"]
                    or digest(original.with_suffix(".npz")) != record["prediction_sha256"]):
                raise ValueError("Unverified sequence reuse")
            reused.append(index)
        else:
            computed.append(index)
    if frames != 69075 or retries != report["observed_pressure_retry_frames"]:
        raise ValueError("Coverage counts changed")
    review = {"passed": True, "proof_sha256": proof_hash, "parent_plan_sha256": proof["parent_plan_sha256"],
              "new_plan_sha256": proof["new_plan_sha256"], "frames": frames, "shots": len(plan["shots"]),
              "reused_indices": reused, "computed_indices": computed, "reused_shots": len(reused),
              "computed_shots": len(computed), "observed_pressure_retry_frames": retries,
              "test_labels_read": False, "audit_labels_read": False, "mast_accuracy_established": False}
    write(run / "reuse_review.json", review)
    report.update(reuse_proof_sha256=proof_hash, reuse_review_sha256=digest(run / "reuse_review.json"),
                  reused_shots=len(reused), computed_shots=len(computed))
    write(run / "mast_export.json", report)
    print(json.dumps({k: v for k, v in review.items() if k not in ("reused_indices", "computed_indices")}), flush=True)


if __name__ == "__main__":
    main()
