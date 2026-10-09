#!/usr/bin/env python3
"""Apply predeclared evidence gates before staging a first transfer baseline."""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.challenge2.reuse_valid_export import digest, read, write


def bound(item):
    path = ROOT / item["path"]
    if digest(path) != item["sha256"]:
        raise ValueError("Predeclared development evidence changed")
    return read(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--check-plan", action="store_true", help="Check declared evidence without staging or accepting the pending export")
    a = p.parse_args()
    plan = read(a.plan)
    for name, expected in plan["review_source_sha256"].items():
        if digest(ROOT / name) != expected:
            raise ValueError("The predeclared review implementation changed")
    if plan.get("first_transfer_baseline_only") is not True:
        raise ValueError("This review does not authorize subsequent transfer experiments")
    run = ROOT / plan["export"]
    if digest(run / "export_plan.json") != plan["export_plan_sha256"]:
        raise ValueError("Frozen transfer policy changed")
    for item in plan["checks"]:
        bound(item)
    manifests = [ROOT / item["path"] for item in plan["query_manifests"]]
    if any(digest(path) != item["sha256"] for path,item in zip(manifests,plan["query_manifests"])) or digest(manifests[0]) != digest(manifests[1]):
        raise ValueError("Development and control query manifests differ")
    state = read(ROOT / "results/candidates/submission_state.json")
    if any(x.get("verified_success") and (x.get("dual") or x.get("scores", {}).get("mast_S", 0) > 0)
           for x in state["entries"].values()):
        raise ValueError("A transfer result already exists; perform a new improvement review")
    best = state["historical_best"]["d3d_S"]
    if best["score"] < .85:
        raise ValueError("The verified DIII-D result does not meet transfer eligibility")
    current = bound(plan["development_physics_report"])
    control = bound(plan["development_control_report"])
    development = bound(plan["development_coverage_report"])
    if (current["audit_labels_read"] or current["mast_labels_read"] or not current["same_queries"]
            or current["frames"] != control["frames"] or current["shots"] != control["shots"]
            or current["frames"] != 1763 or current["shots"] != 8
            or current["scores"]["trial_29"]["S"] < control["scores"]["trial_15"]["S"]-1e-6
            or development["machines"]["DIII-D"]["scalar_valid"] < 1760
            or development["machines"]["DIII-D"]["runtime_limited_frames"]):
        raise ValueError("Corrected development behavior failed the declared control")
    if a.check_plan:
        print(json.dumps({"predeclared_evidence_passed": True, "full_export_review_pending": True,
                          "eligible_for_submission": False}), flush=True)
        return
    coverage_path = run / "complete_coverage_review.json"
    subprocess.run([sys.executable, str(ROOT / "scripts/challenge2/summarize_export_coverage.py"),
                    "--run", str(run), "--output", str(coverage_path)], check=True)
    coverage = read(coverage_path)
    value = coverage["machines"]["MAST"]
    exported, reuse = read(run / "mast_export.json"), read(run / "reuse_review.json")
    if (coverage["scope"] != "full" or not coverage["complete_native_queries"]
            or value["shots"] != 1206 or value["frames"] != 69075 or value["scalar_valid"] != 69075
            or value["finite_positive_scalars"] != 69075 or value["runtime_limited_frames"] or value["exceptions"]
            or coverage["mast_labels_read"] or coverage["audit_labels_read"] or not reuse["passed"]
            or digest(run / "reuse_review.json") != exported["reuse_review_sha256"]
            or digest(run / "reuse_proof.json") != exported["reuse_proof_sha256"]
            or coverage["plan_sha256"] != plan["export_plan_sha256"]):
        raise ValueError("Full native export failed the declared admission checks")
    review = {"created_utc": datetime.now(timezone.utc).isoformat(), "decision": "accept",
              "admission_plan_sha256": digest(a.plan), "purpose": "First complete transfer baseline",
              "development_physics_report": plan["development_physics_report"],
              "mast_coverage_report": {"path": str(coverage_path.relative_to(ROOT)), "sha256": digest(coverage_path)},
              "mast_labels_read": False, "audit_labels_read": False, "mast_accuracy_established": False,
              "rationale": (f"First input-only MAST baseline has all 69,075 native frames validated, finite positive derived scalars "
                  f"and no runtime cutoffs. The corrected held-out DIII-D score remains {current['scores']['trial_29']['S']:.6f}. "
                  f"Strict numerical convergence holds on {value['converged']}/69,075 MAST frames; approximate outputs and "
                  "unverified MAST accuracy are explicitly retained in the evidence. The submitted DIII-D payload must remain "
                  "byte-identical to the best verified official entry. An initial official transfer measurement is worthwhile "
                  "without claiming near-unity accuracy or a verified improvement over an existing transfer entry.")}
    review_path = run / "transfer_review.json"
    write(review_path, review)
    subprocess.run([sys.executable, str(ROOT / "scripts/challenge2/stage_transfer_candidate.py"),
                    "--export", str(run), "--candidate", plan["candidate"], "--review", str(review_path)], check=True)
    print(json.dumps({"candidate": plan["candidate"], "review": "accepted", "periodic_submission_queue": True}), flush=True)


if __name__ == "__main__":
    main()
