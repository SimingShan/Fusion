#!/usr/bin/env python3
"""Match native exported predictions to an independent completed evaluation."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.codabench_submit import write_json


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--export", type=Path, required=True)
    p.add_argument("--evaluation", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--minimum-shots", type=int, default=1)
    a = p.parse_args()
    if a.output.exists():
        raise ValueError("Preserve completed parity checks")
    plan = json.loads((a.export / "export_plan.json").read_text())
    completed = json.loads((a.export / "mast_export.json").read_text())
    evaluated = json.loads((a.evaluation / "summary.json").read_text())
    if completed["plan_sha256"] != digest(a.export / "export_plan.json"):
        raise ValueError("Export plan changed")
    if completed["prediction_sha256"] != digest(a.export / "mast_public_test.npz"):
        raise ValueError("Export payload changed")
    if plan["prior_sha256"] != evaluated["prior_sha256"]:
        raise ValueError("Different inference priors")
    for key in ("max_evaluations", "max_seconds", "pressure_factor", "allow_approximate_seed",
                "retry_observed_pressure", "solver_method", "constrain_pressure_current"):
        default = False if key == "constrain_pressure_current" else None
        if plan.get(key, default) != evaluated["settings"].get(key, default):
            raise ValueError(f"Different prediction setting: {key}")
    if evaluated["settings"]["frames"] != 0:
        raise ValueError("A complete native evaluation is required")
    records = [x for x in evaluated["records"] if x["machine"] == "MAST"]
    by_shot = {}
    for row in records:
        by_shot.setdefault(row["shot"], []).append(row)
    matched = []
    for index, item in enumerate(plan["shots"]):
        name = Path(item["path"]).stem
        if name not in by_shot:
            continue
        rows = by_shot[name]
        receipt = json.loads((a.export / "shots" / f"{index:04d}.json").read_text())
        path = a.export / "shots" / f"{index:04d}.npz"
        if (receipt["status"] != "complete" or receipt["plan_sha256"] != completed["plan_sha256"]
                or digest(path) != receipt["prediction_sha256"]
                or [x["frame"] for x in rows] != list(range(receipt["frames"]))):
            raise ValueError("Incomplete or changed native shot")
        with np.load(path) as arrays:
            for row in rows:
                frame = row["frame"]
                with np.load(a.evaluation / f"MAST_{name}_{frame:04d}.npz") as separate:
                    np.testing.assert_array_equal(arrays["psirz"][frame], separate["psi"])
                for key in ("q95", "betaN"):
                    np.testing.assert_array_equal(arrays[key][frame], np.float32(row[key]))
        matched.append({"shot": name, "frames": len(rows), "receipt_sha256": digest(path.with_suffix(".json"))})
    if len(matched) < a.minimum_shots:
        raise ValueError("Insufficient shared complete shots")
    result = {"passed": True, "prediction_arrays_bit_identical": True, "shots": len(matched),
              "frames": sum(x["frames"] for x in matched), "matched": matched,
              "export_plan_sha256": digest(a.export / "export_plan.json"),
              "export_report_sha256": digest(a.export / "mast_export.json"),
              "evaluation_summary_sha256": digest(a.evaluation / "summary.json"),
              "check_source_sha256": digest(__file__), "mast_labels_read": False, "audit_labels_read": False}
    write_json(a.output, result)
    print({k: v for k, v in result.items() if k != "matched"})


if __name__ == "__main__":
    main()
