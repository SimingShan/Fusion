#!/usr/bin/env python3
"""Summarize complete native inference validity without inferring MAST accuracy."""
from collections import Counter
from pathlib import Path
import argparse
import hashlib
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.codabench_submit import write_json


def distribution(records, key):
    values = np.asarray([np.nan if x.get(key) is None else x[key] for x in records], float)
    finite = values[np.isfinite(values)]
    return {"total": len(values), "finite": len(finite),
            **({"minimum": float(finite.min()), "median": float(np.median(finite)),
                "p95": float(np.quantile(finite, .95)), "maximum": float(finite.max())} if len(finite) else {})}


def summarize(records, seconds):
    return {"shots": len({x["shot"] for x in records}), "frames": len(records),
            "converged": sum(bool(x.get("converged")) for x in records),
            "scalar_valid": sum(bool(x.get("scalar_valid")) for x in records),
            "finite_positive_scalars": sum(all(x.get(key) is not None and np.isfinite(x[key]) and x[key] > 0
                                               for key in ("q95", "betaN")) for x in records),
            "runtime_limited_frames": sum(x.get("solver_elapsed_seconds", 0) >= seconds for x in records),
            "exceptions": sum("error" in x or "failure" in x for x in records),
            "observed_pressure_retry_frames": sum(bool(x.get("pressure_observation_retry_used")) for x in records),
            "statuses": dict(Counter(x.get("status", "missing") for x in records)),
            "statistics": {key: distribution(records, key) for key in
                ("nonlinear_profile_relative_residual", "solver_elapsed_seconds", "Ip_relative_error",
                 "current_negative_mass_fraction", "q95", "betaN", "pressure_relative_rmse",
                 "positive_pressure_outside_surface_fraction")}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Preserve existing coverage reviews")
    source = args.evaluation / "summary.json"
    value = json.loads(source.read_text())
    if (value.get("complete_native_queries") is not True or value["mast_labels_read"]
            or value["audit_labels_read"]):
        raise ValueError("Require complete native queries with permitted target provenance")
    records = value["records"]
    if len({(x["machine"], x["shot"], x["frame"]) for x in records}) != len(records):
        raise ValueError("Duplicate native query")
    machines = sorted({x["machine"] for x in records})
    report = {"evaluation_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
              "prior_sha256": value["prior_sha256"], "source_sha256": value["source_sha256"],
              "complete_native_queries": True, "mast_labels_read": False, "audit_labels_read": False,
              "mast_accuracy_established": False, "eligible_for_submission": False,
              "machines": {name: summarize([x for x in records if x["machine"] == name],
                                              value["settings"]["max_seconds"]) for name in machines},
              "by_shot": {shot: summarize([x for x in records if x["shot"] == shot],
                                              value["settings"]["max_seconds"])
                          for shot in dict.fromkeys(x["shot"] for x in records)},
              "limitations": "Finite outputs and numerical convergence do not establish MAST accuracy",
              "review_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    write_json(args.output, report)
    print(json.dumps({name: {k: v for k, v in report["machines"][name].items() if k != "statistics"}
                      for name in machines}), flush=True)


if __name__ == "__main__":
    main()
