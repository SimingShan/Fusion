#!/usr/bin/env python3
"""Review a complete first transfer entry after independently checked recovery."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.codabench_submit import write_json


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def binding(path):
    path = Path(path)
    return {"path": str(path.relative_to(ROOT)), "sha256": digest(path)}


def bound(item):
    path = ROOT / item["path"]
    if digest(path) != item["sha256"]:
        raise ValueError("Predeclared evidence changed")
    return read(path)


def freeze_plan(path):
    if path.exists():
        raise ValueError("Preserve the predeclared review")
    run = ROOT / "results/challenge2/c2_export_061"
    if path.resolve() != run / "transfer_review_plan.json":
        raise ValueError("This review preparation is specific to the frozen recovery candidate")
    sources = ("scripts/challenge2/review_transfer_recovery.py", "scripts/challenge2/check_recovery_transfer_review.py",
               "scripts/challenge2/summarize_export_coverage.py", "scripts/challenge2/summarize_native_coverage.py",
               "scripts/challenge2/reuse_restart_export.py", "scripts/challenge2/check_fresh_initialization.py",
               "scripts/challenge2/check_positive_current_trials.py", "scripts/challenge2/reuse_valid_export.py",
               "scripts/challenge2/stage_transfer_candidate.py", "scripts/package_dual_submission.py",
               "scripts/periodic_submission.py", "scripts/codabench_submit.py")
    plan = {"created_utc": datetime.now(timezone.utc).isoformat(), "candidate": "20260920-061",
        "first_transfer_baseline_only": True, "export": str(run.relative_to(ROOT)),
        "export_plan_sha256": digest(run / "export_plan.json"),
        "review_source_sha256": {name: digest(ROOT / name) for name in sources},
        "checks": [binding(ROOT / name) for name in (
            "results/checks/c2_fresh_initialization_057.json", "results/checks/c2_restart_reuse_062.json",
            "results/checks/c2_restart_replay_063.json", "results/checks/c2_fresh_export_parity_057.json",
            "results/checks/c2_export_evaluation_parity_044.json")],
        "query_manifests": [binding(ROOT / f"results/challenge2/development_comparison_{n:03}/data/manifest.json") for n in (10, 13)],
        "superseded_export_failure": binding(ROOT / "results/challenge2/c2_export_smoke_050/shots/0000.json"),
        "reuse_proof": binding(run / "reuse_proof.json"),
        "development_evaluation": "results/challenge2/c2_evaluation_059",
        "broad_evaluation": "results/challenge2/c2_evaluation_060",
        "development_physics_report": binding(ROOT / "results/challenge2/development_comparison_013/comparison.json"),
        "previous_development_report": binding(ROOT / "results/challenge2/development_comparison_010/comparison.json"),
        "development_coverage_report": binding(ROOT / "results/challenge2/c2_evaluation_059/coverage_review.json"),
        "broad_coverage_report": binding(ROOT / "results/challenge2/c2_evaluation_060/coverage_review.json"),
        "native_export_checks": [{"run": f"results/challenge2/c2_export_smoke_{n:03}", "frames": frames,
            "report": binding(ROOT / f"results/challenge2/c2_export_smoke_{n:03}/mast_export.json")} for n, frames in ((57, 104), (58, 281), (63, 153))],
        "minimum_original_gain": .0001, "maximum_recovery_regression": .000001,
        "eligible_for_submission": False, "mast_accuracy_established": False,
        "mast_labels_read": False, "audit_labels_read": False}
    for name in sources:
        destination = run / "admission_snapshot" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, destination)
    write_json(path, plan)


def check_inference_sources(run, evaluation):
    plan, summary = read(run / "export_plan.json"), read(evaluation / "summary.json")
    parents = summary.get("parents_sha256", {})
    if (not parents or not summary.get("complete_native_queries") or summary["mast_labels_read"]
            or summary["audit_labels_read"] or summary["prior_sha256"] != plan["prior_sha256"]):
        raise ValueError("Complete matching evaluation provenance required")
    keys = ("solver_method", "max_seconds", "max_evaluations", "pressure_factor", "allow_approximate_seed",
            "retry_observed_pressure", "retry_fresh_initialization", "constrain_pressure_current")
    if any(plan[k] != summary["settings"][k] for k in keys) or summary["settings"].get("positive_current_trials"):
        raise ValueError("Evaluated inference settings differ from the export")
    for name, expected in parents.items():
        path = ROOT / name
        if digest(path) != expected:
            raise ValueError("Evaluation shard evidence changed")
        shard = read(path)
        if shard["source_sha256"] != summary["source_sha256"] or shard["prior_sha256"] != plan["prior_sha256"]:
            raise ValueError("Evaluation shard policy differs")
        for name, expected in summary["source_sha256"].items():
            if (plan["source_sha256"].get(name) != expected
                    or digest(path.parent / "source_snapshot" / name) != expected):
                raise ValueError("Exported numerical source differs from evaluation")
    return summary


def check_plan(path):
    plan = read(path)
    if (plan.get("first_transfer_baseline_only") is not True or plan.get("minimum_original_gain") != .0001
            or plan.get("maximum_recovery_regression") != .000001):
        raise ValueError("The predeclared first-transfer scope or thresholds changed")
    for name, expected in plan["review_source_sha256"].items():
        if digest(ROOT / name) != expected:
            raise ValueError("The predeclared review implementation changed")
    run = ROOT / plan["export"]
    if digest(run / "export_plan.json") != plan["export_plan_sha256"]:
        raise ValueError("Frozen export policy changed")
    exported = read(run / "export_plan.json")
    if (exported["scope"] != "full" or len(exported["shots"]) != 1206
            or not exported.get("retry_fresh_initialization") or not exported.get("retry_observed_pressure")
            or exported.get("constrain_pressure_current") or exported["test_labels_read"] or exported["audit_labels_read"]
            or digest(run / "checkpoints/prior.joblib") != exported["prior_sha256"]):
        raise ValueError("Unexpected export scope or predictor")
    for name, expected in exported["source_sha256"].items():
        if digest(run / "source_snapshot" / name) != expected:
            raise ValueError("Frozen predictor source changed")
    for item in plan["checks"]:
        if bound(item).get("passed") is not True:
            raise ValueError("An independent check failed")
    hashes = [digest(ROOT / item["path"]) for item in plan["query_manifests"]]
    if hashes != [item["sha256"] for item in plan["query_manifests"]] or len(set(hashes)) != 1:
        raise ValueError("Development and control query manifests differ")
    if bound(plan["superseded_export_failure"]).get("status") != "failed":
        raise ValueError("Declared parent sequence failure is absent")
    proof = bound(plan["reuse_proof"])
    if (not proof["passed"] or proof["new_plan_sha256"] != plan["export_plan_sha256"]
            or Path(proof["run"]) != run or proof["parity_frames"] != 104
            or proof["corrected_frames"] != 281 or proof["initialization_retry_frames"] < 1):
        raise ValueError("Complete parity and recovery reuse proof required")
    for name, expected in proof["helper_sha256"].items():
        if digest(Path(proof["source_root"]) / name) != expected:
            raise ValueError("Reuse implementation changed")
    for item in proof["evidence"]:
        if digest(item["path"]) != item["sha256"]:
            raise ValueError("Reuse evidence changed")
    development_summary = check_inference_sources(run, ROOT / plan["development_evaluation"])
    broad_summary = check_inference_sources(run, ROOT / plan["broad_evaluation"])
    for item in plan["native_export_checks"]:
        report, smoke = bound(item["report"]), read(ROOT / item["run"] / "export_plan.json")
        if (report["status"] != "smoke_complete" or report["frames"] != item["frames"]
                or report["scalar_valid"] != item["frames"] or report["runtime_limited_frames"]
                or report["source_sha256"] != exported["source_sha256"] or report["prior_sha256"] != exported["prior_sha256"]
                or report["plan_sha256"] != digest(ROOT / item["run"] / "export_plan.json")
                or report["prediction_sha256"] != digest(ROOT / item["run"] / "mast_public_test.npz")
                or any(smoke[k] != exported[k] for k in ("solver_method", "constrain_pressure_current", "max_seconds",
                    "max_evaluations", "pressure_factor", "allow_approximate_seed", "retry_observed_pressure", "retry_fresh_initialization"))):
            raise ValueError("Complete-sequence check differs or failed")
    state = read(ROOT / "results/candidates/submission_state.json")
    if any(x.get("verified_success") and (x.get("dual") or x.get("scores", {}).get("mast_S", 0) > 0)
           for x in state["entries"].values()):
        raise ValueError("A transfer result exists; a new improvement review is required")
    if state["historical_best"]["d3d_S"]["score"] < .85:
        raise ValueError("Verified DIII-D entry is ineligible")
    comparison, previous = bound(plan["development_physics_report"]), bound(plan["previous_development_report"])
    current, control = comparison["scores"]["trial_59"]["S"], comparison["scores"]["trial_36"]["S"]
    original = previous["scores"]["trial_29"]["S"]
    development, broad = bound(plan["development_coverage_report"]), bound(plan["broad_coverage_report"])
    for evidence, evaluation in ((development, plan["development_evaluation"]), (broad, plan["broad_evaluation"])):
        if evidence["evaluation_sha256"] != digest(ROOT / evaluation / "summary.json"):
            raise ValueError("Coverage is not bound to the evaluated prediction set")
    dev, mast = development["machines"]["DIII-D"], broad["machines"]["MAST"]
    if (any(x["audit_labels_read"] or x["mast_labels_read"] or not x["same_queries"]
                or x["shots"] != 8 or x["frames"] != 1763 for x in (comparison, previous))
            or control != previous["scores"]["trial_36"]["S"] or control < original+.0001
            or current < control-.000001 or current < original+.0001
            or dev["frames"] != 1763 or dev["scalar_valid"] < 1760 or dev["runtime_limited_frames"] or dev["exceptions"]
            or mast["shots"] != 32 or mast["frames"] != 1642 or mast["finite_positive_scalars"] != 1642
            or mast["scalar_valid"] != 1642 or mast["runtime_limited_frames"] or mast["exceptions"]):
        raise ValueError("The declared accuracy or native validity requirements failed")
    return plan, run, {"current": current, "control": control, "original": original}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--freeze-plan", action="store_true")
    p.add_argument("--check-plan", action="store_true")
    a = p.parse_args()
    if a.freeze_plan:
        freeze_plan(a.plan)
    plan, run, scores = check_plan(a.plan)
    if a.check_plan:
        print(json.dumps({"predeclared_evidence_passed": True, "full_export_review_pending": True,
                          "eligible_for_submission": False, **scores}), flush=True)
        return
    exported, reuse = read(run / "mast_export.json"), read(run / "reuse_review.json")
    if (not reuse["passed"] or reuse["new_plan_sha256"] != plan["export_plan_sha256"]
            or reuse["frames"] != 69075 or reuse["shots"] != 1206
            or exported.get("reuse_proof_sha256") != plan["reuse_proof"]["sha256"]
            or exported.get("reuse_review_sha256") != digest(run / "reuse_review.json")):
        raise ValueError("Full export reuse provenance is incomplete")
    coverage_path = run / "admission_coverage_review.json"
    subprocess.run([sys.executable, str(ROOT / "scripts/challenge2/summarize_export_coverage.py"),
                    "--run", str(run), "--output", str(coverage_path)], check=True)
    coverage = read(coverage_path)
    value = coverage["machines"]["MAST"]
    if (coverage["scope"] != "full" or not coverage["complete_native_queries"]
            or value["shots"] != 1206 or value["frames"] != 69075 or value["scalar_valid"] != 69075
            or value["finite_positive_scalars"] != 69075 or value["runtime_limited_frames"] or value["exceptions"]
            or coverage["mast_labels_read"] or coverage["audit_labels_read"] or coverage["plan_sha256"] != plan["export_plan_sha256"]):
        raise ValueError("Full native export failed independent admission")
    review = {"created_utc": datetime.now(timezone.utc).isoformat(), "decision": "accept",
        "admission_plan_sha256": digest(a.plan), "purpose": "First complete transfer baseline",
        "development_physics_report": plan["development_physics_report"], "mast_coverage_report": binding(coverage_path),
        "mast_labels_read": False, "audit_labels_read": False, "mast_accuracy_established": False,
        "rationale": (f"The parent failed a complete native sequence. The conditional recovery passed all known sequences "
            f"and preserves valid parent outputs under checked sequence reuse. Held-out DIII-D S is {scores['current']:.6f} "
            f"versus {scores['control']:.6f} for its direct control and {scores['original']:.6f} for the earlier baseline. "
            f"All 69,075 native MAST outputs passed independent validity and coverage checks without runtime cutoffs. "
            f"Strict convergence holds on {value['converged']} frames; hidden-target accuracy is unverified. "
            "The first official transfer measurement preserves the exact best verified DIII-D prediction payload.")}
    review_path = run / "transfer_review.json"
    if review_path.exists():
        raise ValueError("Preserve the prior admission decision")
    write_json(review_path, review)
    subprocess.run([sys.executable, str(ROOT / "scripts/challenge2/stage_transfer_candidate.py"),
                    "--export", str(run), "--candidate", plan["candidate"], "--review", str(review_path)], check=True)
    print(json.dumps({"candidate": plan["candidate"], "review": "accepted", "periodic_submission_queue": True}), flush=True)


if __name__ == "__main__":
    main()
