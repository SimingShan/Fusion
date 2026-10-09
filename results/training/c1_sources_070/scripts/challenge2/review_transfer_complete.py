#!/usr/bin/env python3
"""Review a complete first transfer baseline with improved development evidence."""
import argparse
import ast
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.codabench_submit import write_json
from scripts.challenge2.check_current_constrained_pressure import RemoveDisabledOption


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def bound(item):
    path = ROOT / item["path"]
    if digest(path) != item["sha256"]:
        raise ValueError("Predeclared evidence changed")
    return read(path)


def check_inference_sources(export, evaluation):
    plan, summary = read(export / "export_plan.json"), read(evaluation / "summary.json")
    parents = summary.get("parents_sha256", {})
    if not parents or any(digest(ROOT / name) != expected for name, expected in parents.items()):
        raise ValueError("Complete evaluation shard provenance changed")
    source_snapshot = (ROOT / next(iter(parents))).parent / "source_snapshot"
    if plan["prior_sha256"] != summary["prior_sha256"] or plan.get("constrain_pressure_current"):
        raise ValueError("The evaluated inference policy differs from the export")
    for key in ("solver_method", "max_seconds", "max_evaluations", "pressure_factor", "allow_approximate_seed", "retry_observed_pressure"):
        if plan[key] != summary["settings"][key]:
            raise ValueError(f"The evaluated setting differs: {key}")
    for name, expected in summary["source_sha256"].items():
        if name == "scripts/challenge2/evaluate_profile_prior.py":
            old = ast.parse((source_snapshot / name).read_text())
            if digest(source_snapshot / name) != expected:
                raise ValueError("Evaluation source changed")
            new = ast.parse((export / "source_snapshot" / name).read_text())
            for function in ("infer", "infer_with_retry", "valid_prediction"):
                before = next(x for x in old.body if isinstance(x, ast.FunctionDef) and x.name == function)
                after = next(x for x in new.body if isinstance(x, ast.FunctionDef) and x.name == function)
                if ast.dump(before) != ast.dump(RemoveDisabledOption().visit(copy.deepcopy(after))):
                    raise ValueError("The exported numerical function differs from development")
        elif plan["source_sha256"].get(name) != expected:
            raise ValueError("The exported numerical dependency differs from development")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--check-plan", action="store_true")
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
    exported_plan = read(run / "export_plan.json")
    for name, expected in exported_plan["source_sha256"].items():
        if digest(run / "source_snapshot" / name) != expected:
            raise ValueError("Frozen predictor source changed")
    if digest(run / "checkpoints/prior.joblib") != exported_plan["prior_sha256"]:
        raise ValueError("Frozen inference prior changed")
    for item in plan["checks"]:
        if bound(item).get("passed") is not True:
            raise ValueError("A required independent check did not pass")
    manifests = [ROOT / item["path"] for item in plan["query_manifests"]]
    if (any(digest(path) != item["sha256"] for path,item in zip(manifests,plan["query_manifests"]))
            or digest(manifests[0]) != digest(manifests[1])):
        raise ValueError("Development and control query manifests differ")
    previous_failure = bound(plan["superseded_export_failure"])
    if previous_failure.get("status") != "failed":
        raise ValueError("The declared superseded export failure is absent")
    check_inference_sources(run, ROOT / plan["development_evaluation"])
    check_inference_sources(run, ROOT / plan["broad_evaluation"])
    for item in plan["native_export_checks"]:
        report = bound(item["report"])
        smoke = read(ROOT / item["run"] / "export_plan.json")
        if (report["status"] != "smoke_complete" or report["frames"] != item["frames"]
                or report["scalar_valid"] != item["frames"] or report["runtime_limited_frames"]
                or report["source_sha256"] != exported_plan["source_sha256"]
                or report["prior_sha256"] != exported_plan["prior_sha256"]
                or report["plan_sha256"] != digest(ROOT / item["run"] / "export_plan.json")
                or report["prediction_sha256"] != digest(ROOT / item["run"] / "mast_public_test.npz")
                or any(smoke[k] != exported_plan[k] for k in ("solver_method", "constrain_pressure_current", "max_seconds",
                    "max_evaluations", "pressure_factor", "allow_approximate_seed", "retry_observed_pressure"))):
            raise ValueError("A complete-sequence export check changed or failed")
    state = read(ROOT / "results/candidates/submission_state.json")
    if any(x.get("verified_success") and (x.get("dual") or x.get("scores", {}).get("mast_S", 0) > 0)
           for x in state["entries"].values()):
        raise ValueError("A transfer result already exists; perform a new improvement review")
    if state["historical_best"]["d3d_S"]["score"] < .85:
        raise ValueError("Verified DIII-D score is ineligible for transfer")
    comparison = bound(plan["development_physics_report"])
    current, control = comparison["scores"]["trial_36"]["S"], comparison["scores"]["trial_29"]["S"]
    development = bound(plan["development_coverage_report"])["machines"]["DIII-D"]
    broad = bound(plan["broad_coverage_report"])["machines"]["MAST"]
    if (comparison["audit_labels_read"] or comparison["mast_labels_read"] or not comparison["same_queries"]
            or comparison["shots"] != 8 or comparison["frames"] != 1763 or current < control+.0001
            or development["frames"] != 1763 or development["scalar_valid"] < 1760
            or development["runtime_limited_frames"] or development["exceptions"]
            or broad["shots"] != 32 or broad["frames"] != 1642 or broad["finite_positive_scalars"] != 1642
            or broad["runtime_limited_frames"] or broad["exceptions"]):
        raise ValueError("The declared development evidence failed")
    if a.check_plan:
        print(json.dumps({"predeclared_evidence_passed": True, "full_export_review_pending": True,
                          "eligible_for_submission": False}), flush=True)
        return
    coverage_path = run / "admission_coverage_review.json"
    subprocess.run([sys.executable, str(ROOT / "scripts/challenge2/summarize_export_coverage.py"),
                    "--run", str(run), "--output", str(coverage_path)], check=True)
    coverage = read(coverage_path)
    value = coverage["machines"]["MAST"]
    if (coverage["scope"] != "full" or not coverage["complete_native_queries"]
            or value["shots"] != 1206 or value["frames"] != 69075 or value["scalar_valid"] != 69075
            or value["finite_positive_scalars"] != 69075 or value["runtime_limited_frames"] or value["exceptions"]
            or coverage["mast_labels_read"] or coverage["audit_labels_read"]
            or coverage["plan_sha256"] != plan["export_plan_sha256"]):
        raise ValueError("Full native export failed admission checks")
    review = {"created_utc": datetime.now(timezone.utc).isoformat(), "decision": "accept",
        "admission_plan_sha256": digest(a.plan), "purpose": "First complete transfer baseline",
        "development_physics_report": plan["development_physics_report"],
        "mast_coverage_report": {"path": str(coverage_path.relative_to(ROOT)), "sha256": digest(coverage_path)},
        "mast_labels_read": False, "audit_labels_read": False, "mast_accuracy_established": False,
        "rationale": (f"The earlier frozen export failed an entire-sequence check. This candidate passed all such checks "
            f"and improves the same 1,763-frame held-out DIII-D score from {control:.6f} to {current:.6f}. "
            f"All 69,075 native MAST outputs are validated and have finite positive derived scalars without runtime cutoffs. "
            f"Strict convergence holds on {value['converged']} exported frames. The broader MAST comparison had fewer "
            "strictly converged frames than the control, so hidden-target accuracy remains unverified. "
            "A first official transfer measurement is justified while preserving the best verified DIII-D payload exactly.")}
    review_path = run / "transfer_review.json"
    write_json(review_path, review)
    subprocess.run([sys.executable, str(ROOT / "scripts/challenge2/stage_transfer_candidate.py"),
                    "--export", str(run), "--candidate", plan["candidate"], "--review", str(review_path)], check=True)
    print(json.dumps({"candidate": plan["candidate"], "review": "accepted", "periodic_submission_queue": True}), flush=True)


if __name__ == "__main__":
    main()
