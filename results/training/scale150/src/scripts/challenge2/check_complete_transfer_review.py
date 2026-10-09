#!/usr/bin/env python3
"""Check pending-review admission and rejection without staging a candidate."""
from pathlib import Path
import argparse
import copy
import json
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.challenge2.reuse_valid_export import digest, read, write


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise ValueError("Preserve previous checks")
    command = [sys.executable, str(ROOT / "scripts/challenge2/review_transfer_complete.py")]
    valid = subprocess.run(command + ["--plan", str(a.plan), "--check-plan"], capture_output=True, text=True, check=True)
    pending = json.loads(valid.stdout)
    if not pending["full_export_review_pending"] or pending["eligible_for_submission"]:
        raise ValueError("A pending full export was accepted")
    checks = {"actual_development_evidence_passed": True, "pending_export_not_accepted": True}
    plan = read(a.plan)
    changes = [("changed_export_policy_rejected", lambda x: x.update(export_plan_sha256="0"*64)),
               ("changed_development_evidence_rejected", lambda x: x["development_physics_report"].update(sha256="0"*64)),
               ("changed_queries_rejected", lambda x: x["query_manifests"][0].update(sha256="0"*64)),
               ("noninitial_transfer_scope_rejected", lambda x: x.update(first_transfer_baseline_only=False)),
               ("changed_broad_evidence_rejected", lambda x: x["broad_coverage_report"].update(sha256="0"*64)),
               ("mismatched_evaluated_policy_rejected", lambda x: x.update(development_evaluation="results/challenge2/c2_evaluation_029"))]
    with tempfile.TemporaryDirectory() as directory:
        for name, change in changes:
            altered = copy.deepcopy(plan)
            change(altered)
            path = Path(directory) / "altered_plan.json"
            write(path, altered)
            result = subprocess.run(command + ["--plan", str(path), "--check-plan"], capture_output=True, text=True)
            if result.returncode == 0:
                raise ValueError(f"An invalid review contract was accepted: {name}")
            checks[name] = True
        output = Path(directory) / "incomplete_coverage.json"
        result = subprocess.run([sys.executable, str(ROOT / "scripts/challenge2/summarize_export_coverage.py"),
            "--run", str(ROOT / "results/challenge2/c2_export_smoke_027"), "--output", str(output)], capture_output=True, text=True)
        if result.returncode == 0 or output.exists():
            raise ValueError("A smoke export was accepted as full coverage")
        checks["incomplete_native_export_rejected"] = True
    report = {"passed": True, "checks": checks, "review_plan_sha256": digest(a.plan),
              "submission_attempted": False, "candidate_staged": False, "test_labels_read": False, "audit_labels_read": False}
    write(a.output, report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
