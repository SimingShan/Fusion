#!/usr/bin/env python3
"""Exercise the independent recovery admission without staging or submitting."""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.challenge2.review_transfer_recovery import check_plan, digest
from scripts.codabench_submit import write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise ValueError("Preserve previous checks")
    check_plan(a.plan)
    command = [sys.executable, str(ROOT / "scripts/challenge2/review_transfer_recovery.py")]
    pending = json.loads(subprocess.check_output(command + ["--plan", str(a.plan), "--check-plan"], text=True))
    if not pending["full_export_review_pending"] or pending["eligible_for_submission"]:
        raise ValueError("Pending export was accepted")
    checks = {"actual_development_evidence_passed": True, "pending_full_export_not_accepted": True}
    plan = json.loads(a.plan.read_text())
    changes = [
        ("changed_export_policy_rejected", lambda x: x.update(export_plan_sha256="0"*64)),
        ("changed_development_evidence_rejected", lambda x: x["development_physics_report"].update(sha256="0"*64)),
        ("changed_queries_rejected", lambda x: x["query_manifests"][0].update(sha256="0"*64)),
        ("noninitial_transfer_scope_rejected", lambda x: x.update(first_transfer_baseline_only=False)),
        ("changed_broad_evidence_rejected", lambda x: x["broad_coverage_report"].update(sha256="0"*64)),
        ("mismatched_evaluation_policy_rejected", lambda x: x.update(development_evaluation="results/challenge2/c2_evaluation_036")),
        ("changed_reuse_proof_rejected", lambda x: x["reuse_proof"].update(sha256="0"*64)),
        ("relaxed_accuracy_threshold_rejected", lambda x: x.update(maximum_recovery_regression=.1)),
        ("changed_smoke_evidence_rejected", lambda x: x["native_export_checks"][0]["report"].update(sha256="0"*64)),
    ]
    with tempfile.TemporaryDirectory(prefix="c2_review_", dir=ROOT / "results/checks") as tmp:
        for name, change in changes:
            altered = copy.deepcopy(plan)
            change(altered)
            path = Path(tmp) / "altered_plan.json"
            write_json(path, altered)
            try:
                check_plan(path)
            except (ValueError, KeyError):
                checks[name] = True
            else:
                raise ValueError(f"Unsafe review accepted: {name}")
        output = Path(tmp) / "incomplete_coverage.json"
        result = subprocess.run([sys.executable, str(ROOT / "scripts/challenge2/summarize_export_coverage.py"),
            "--run", str(ROOT / "results/challenge2/c2_export_smoke_058"), "--output", str(output)], capture_output=True, text=True)
        if result.returncode == 0 or output.exists():
            raise ValueError("Incomplete export was accepted as full coverage")
        checks["incomplete_export_rejected"] = True
    report = {"passed": True, "checks": checks, "review_plan_sha256": digest(a.plan),
              "submission_attempted": False, "candidate_staged": False,
              "test_labels_read": False, "audit_labels_read": False}
    write_json(a.output, report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
