#!/usr/bin/env python3
"""Bind and reuse whole valid sequences under a validity-only restart policy."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.challenge2.reuse_valid_export import checked_plan, checked_smoke, digest, read, write
from scripts.challenge2.check_fresh_initialization import prove_unchanged


def compatible_plans(parent, run):
    old, new = checked_plan(parent), checked_plan(run)
    ignored = {"created_utc", "source_sha256", "retry_fresh_initialization"}
    if ({k: v for k, v in old.items() if k not in ignored}
            != {k: v for k, v in new.items() if k not in ignored}):
        raise ValueError("Plans differ beyond the conditional initialization recovery")
    if (old.get("retry_fresh_initialization", False) or new.get("retry_fresh_initialization") is not True
            or old.get("retry_observed_pressure") is not True or new.get("retry_observed_pressure") is not True
            or old.get("constrain_pressure_current") or new.get("constrain_pressure_current")):
        raise ValueError("Unexpected original or recovered prediction policy")
    changed = {k for k in set(old["source_sha256"]) | set(new["source_sha256"])
               if old["source_sha256"].get(k) != new["source_sha256"].get(k)}
    allowed = {"fusion/mast_physics.py", "scripts/challenge2/evaluate_profile_prior.py",
               "scripts/challenge2/export_mast_predictions.py"}
    if changed != allowed:
        raise ValueError("Unexpected source dependency changes")
    checks = prove_unchanged(parent / "source_snapshot", run / "source_snapshot")
    return old, new, checks


def prepare(a):
    parent, run = a.parent.resolve(), a.run.resolve()
    if (run / "reuse_proof.json").exists() or (run / "reuse/source_snapshot").exists():
        raise ValueError("Preserve frozen reuse proof")
    old, new, checks = compatible_plans(parent, run)
    before, after, correction = [p.resolve() for p in (a.parity_original, a.parity_updated, a.correction)]
    contracts = [checked_smoke(p) for p in (before, after, correction)]
    settings = ("prior_sha256", "packages", "inventory_sha256", "solver_method", "constrain_pressure_current",
                "max_evaluations", "max_seconds", "pressure_factor", "allow_approximate_seed", "retry_observed_pressure")
    for i, (plan, report) in enumerate(contracts):
        expected = old if i == 0 else new
        if (plan["source_sha256"] != expected["source_sha256"]
                or any(plan[k] != expected[k] for k in settings)
                or plan.get("retry_fresh_initialization", False) != expected.get("retry_fresh_initialization", False)):
            raise ValueError("Smoke checks differ from their frozen inference policy")
    old_plan, old_report = contracts[0]
    new_plan, new_report = contracts[1]
    fixed_plan, fixed_report = contracts[2]
    if (old_plan["shots"] != new_plan["shots"]
            or old_report["prediction_sha256"] != new_report["prediction_sha256"]
            or new_report["initialization_retry_frames"] != 0
            or fixed_report["initialization_retry_frames"] < 1):
        raise ValueError("Require exact parity and an actually recovered complete sequence")
    # Freeze the runner and all its local imports; later edits in the workspace
    # cannot silently alter an already admitted reuse decision.
    source = run / "reuse/source_snapshot"
    shutil.copytree(run / "source_snapshot", source)
    helpers = ["scripts/challenge2/reuse_restart_export.py", "scripts/challenge2/reuse_valid_export.py",
               "scripts/challenge2/check_fresh_initialization.py", "scripts/challenge2/check_positive_current_trials.py"]
    for name in helpers:
        destination = source / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, destination)
    runner = source / "scripts/challenge2/reuse_restart_export.py"
    evidence = [{"path": str(p), "sha256": digest(p)} for base in (before, after, correction)
                for p in (base / "export_plan.json", base / "mast_export.json", base / "mast_public_test.npz")]
    proof = {"created_utc": datetime.now(timezone.utc).isoformat(), "passed": True,
             "parent": str(parent), "run": str(run), "parent_plan_sha256": digest(parent / "export_plan.json"),
             "new_plan_sha256": digest(run / "export_plan.json"), "runner": str(runner),
             "runner_sha256": digest(runner), "source_root": str(source),
             "helper_sha256": {str(p.relative_to(source)): digest(p) for p in sorted(source.rglob("*")) if p.is_file()},
             "checks": checks, "evidence": evidence, "parity_frames": new_report["frames"],
             "corrected_frames": fixed_report["frames"], "initialization_retry_frames": fixed_report["initialization_retry_frames"],
             "test_labels_read": False, "audit_labels_read": False,
             "rule": "Preserve complete valid parent sequences including their existing conditional pressure retries; recompute every other sequence from frame zero"}
    write(run / "reuse_proof.json", proof)
    print(json.dumps({k: proof[k] for k in ("passed", "parity_frames", "corrected_frames", "initialization_retry_frames")}), flush=True)


def check_proof(run, require_runner=True):
    proof = read(run / "reuse_proof.json")
    if (not proof["passed"] or Path(proof["run"]) != run
            or digest(proof["runner"]) != proof["runner_sha256"]
            or (require_runner and Path(__file__).resolve() != Path(proof["runner"]))):
        raise ValueError("Execute the frozen checked reuse runner")
    for name, expected in proof["helper_sha256"].items():
        if digest(Path(proof["source_root"]) / name) != expected:
            raise ValueError("A frozen reuse dependency changed")
    parent = Path(proof["parent"])
    if (digest(parent / "export_plan.json") != proof["parent_plan_sha256"]
            or digest(run / "export_plan.json") != proof["new_plan_sha256"]):
        raise ValueError("A prediction plan changed")
    for item in proof["evidence"]:
        if digest(item["path"]) != item["sha256"]:
            raise ValueError("Reuse evidence changed")
    old, new, _ = compatible_plans(parent, run)
    return proof, parent, old, new


def validate_sequence(parent, plan, index, record, plan_hash):
    import pyarrow.parquet as pq
    item = plan["shots"][index]
    source = parent / "shots" / f"{index:04d}.npz"
    if (record["status"] != "complete" or record["index"] != index or record["raw_path"] != item["path"]
            or record["plan_sha256"] != plan_hash or record["scalar_valid"] != record["frames"]
            or record["runtime_limited_frames"] or record.get("initialization_retry_frames", 0)
            or record["prediction_sha256"] != digest(source)):
        raise ValueError("Parent sequence is not eligible for unchanged reuse")
    raw = Path(plan["raw_root"]) / item["path"]
    if raw.stat().st_size != item["size"]:
        raise ValueError("Pinned native input changed")
    times = np.asarray(pq.read_table(raw, columns=["efit_times"])["efit_times"][0].as_py(), dtype="<f8")
    n, frames = len(times), record["records"]
    if (n != record["frames"] or not np.isfinite(times).all() or np.any(np.diff(times) <= 0)
            or hashlib.sha256(times.tobytes()).hexdigest() != record["native_times_sha256"] or len(frames) != n):
        raise ValueError("Native frame coverage changed")
    if any(x["frame"] != i or x["time_ms"] != times[i] or not x["scalar_valid"]
           or not np.isfinite(x["solver_elapsed_seconds"]) or x["solver_elapsed_seconds"] >= plan["max_seconds"]
           or x.get("initialization_retry_used") for i, x in enumerate(frames)):
        raise ValueError("A parent frame is invalid or reached its runtime cutoff")
    if (record["observed_pressure_retry_frames"] != sum(bool(x.get("pressure_observation_retry_used")) for x in frames)
            or record["converged"] != sum(bool(x.get("converged")) for x in frames)):
        raise ValueError("Parent sequence counters differ from its frame evidence")
    with np.load(source, allow_pickle=False) as arrays:
        if set(arrays.files) != {"psirz", "q95", "betaN"}:
            raise ValueError("Unexpected prediction contents")
        for key, shape in (("psirz", (n, 65, 65)), ("q95", (n,)), ("betaN", (n,))):
            value = arrays[key]
            if value.shape != shape or value.dtype != np.float32 or not np.isfinite(value).all():
                raise ValueError("Invalid prediction array")
            if key != "psirz" and (np.any(value <= 0) or not np.array_equal(value, np.asarray([x[key] for x in frames], np.float32))):
                raise ValueError("Invalid or inconsistent scalar array")


def adopt(run):
    proof, parent, old, new = check_proof(run)
    proof_hash = digest(run / "reuse_proof.json")
    reused, skipped = [], []
    with (run / "reuse/adopt.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for index in range(len(old["shots"])):
            original = parent / "shots" / f"{index:04d}.json"
            target = run / "shots" / original.name
            if target.exists() and read(target).get("status") == "complete":
                existing = read(target)
                if existing["plan_sha256"] != proof["new_plan_sha256"] or digest(target.with_suffix(".npz")) != existing["prediction_sha256"]:
                    raise ValueError("A completed child sequence changed")
                continue
            if not original.exists() or read(original).get("status") != "complete":
                skipped.append(index)
                continue
            record, original_hash = read(original), digest(original)
            if record["runtime_limited_frames"]:
                skipped.append(index)
                continue
            validate_sequence(parent, old, index, record, proof["parent_plan_sha256"])
            target.parent.mkdir(exist_ok=True)
            temporary = target.with_name(target.stem + f".{os.getpid()}.tmp.npz")
            shutil.copyfile(original.with_suffix(".npz"), temporary)
            if digest(temporary) != record["prediction_sha256"] or digest(original) != original_hash:
                raise ValueError("Parent output changed during adoption")
            temporary.replace(target.with_suffix(".npz"))
            record.update(plan_sha256=proof["new_plan_sha256"], initialization_retry_frames=0,
                reuse={"type": "unchanged_valid_conditional_sequence", "proof_sha256": proof_hash,
                       "parent_plan_sha256": proof["parent_plan_sha256"], "parent_receipt": str(original),
                       "parent_receipt_sha256": original_hash, "prediction_recomputed": False})
            for frame in record["records"]:
                frame.update(initialization_retry_used=None, original_continuation_failure=None, initialization_retry_failure=None)
            write(target, record)
            reused.append({"index": index, "frames": record["frames"], "prediction_sha256": record["prediction_sha256"]})
        report = {"created_utc": datetime.now(timezone.utc).isoformat(), "proof_sha256": proof_hash,
                  "reused_this_pass": reused, "reused_shots": len(reused), "reused_frames": sum(x["frames"] for x in reused),
                  "indices_requiring_prediction": skipped, "test_labels_read": False, "audit_labels_read": False}
        write(run / "reuse" / f"adoption_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')}.json", report)
    print(json.dumps({"reused_shots": len(reused), "reused_frames": report["reused_frames"], "remaining_sequences": len(skipped)}), flush=True)


def finalize(run):
    proof, parent, old, new = check_proof(run)
    proof_hash, report = digest(run / "reuse_proof.json"), read(run / "mast_export.json")
    if (new["scope"] != "full" or report["status"] != "complete" or report["shots"] != 1206
            or report["frames"] != 69075 or report["scalar_valid"] != 69075 or report["runtime_limited_frames"]
            or report["plan_sha256"] != proof["new_plan_sha256"]
            or digest(run / "mast_public_test.npz") != report["prediction_sha256"]):
        raise ValueError("Complete unchanged native export required")
    reused, computed, frames, recoveries, pressure_retries = [], [], 0, 0, 0
    for index, item in enumerate(new["shots"]):
        path = run / "shots" / f"{index:04d}.json"
        record = read(path)
        if (record["status"] != "complete" or record["index"] != index or record["raw_path"] != item["path"]
                or record["plan_sha256"] != proof["new_plan_sha256"] or record["frames"] != record["scalar_valid"]
                or record["runtime_limited_frames"] or digest(path.with_suffix(".npz")) != record["prediction_sha256"]):
            raise ValueError("Incomplete or changed sequence")
        frames += record["frames"]
        recoveries += record["initialization_retry_frames"]
        pressure_retries += record["observed_pressure_retry_frames"]
        if "reuse" in record:
            reuse, original = record["reuse"], parent / "shots" / path.name
            before = read(original)
            if (reuse["proof_sha256"] != proof_hash or reuse["parent_plan_sha256"] != proof["parent_plan_sha256"]
                    or Path(reuse["parent_receipt"]) != original or digest(original) != reuse["parent_receipt_sha256"]
                    or reuse["prediction_recomputed"] is not False or record["initialization_retry_frames"]
                    or before["prediction_sha256"] != record["prediction_sha256"]
                    or before["observed_pressure_retry_frames"] != record["observed_pressure_retry_frames"]
                    or digest(original.with_suffix(".npz")) != record["prediction_sha256"]):
                raise ValueError("Unverified or changed sequence reuse")
            reused.append(index)
        else:
            computed.append(index)
    if (frames != 69075 or recoveries != report["initialization_retry_frames"]
            or pressure_retries != report["observed_pressure_retry_frames"]):
        raise ValueError("Coverage counters changed")
    review = {"passed": True, "proof_sha256": proof_hash, "parent_plan_sha256": proof["parent_plan_sha256"],
              "new_plan_sha256": proof["new_plan_sha256"], "frames": frames, "shots": len(new["shots"]),
              "reused_indices": reused, "computed_indices": computed, "reused_shots": len(reused),
              "computed_shots": len(computed), "initialization_retry_frames": recoveries,
              "observed_pressure_retry_frames": pressure_retries, "test_labels_read": False,
              "audit_labels_read": False, "mast_accuracy_established": False}
    write(run / "reuse_review.json", review)
    report.update(reuse_proof_sha256=proof_hash, reuse_review_sha256=digest(run / "reuse_review.json"),
                  reused_shots=len(reused), computed_shots=len(computed))
    write(run / "mast_export.json", report)
    print(json.dumps({k: v for k, v in review.items() if k not in ("reused_indices", "computed_indices")}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("prepare", "adopt", "finalize"))
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--parent", type=Path)
    p.add_argument("--parity-original", type=Path)
    p.add_argument("--parity-updated", type=Path)
    p.add_argument("--correction", type=Path)
    a = p.parse_args()
    if a.action == "prepare":
        if any(getattr(a, key) is None for key in ("parent", "parity_original", "parity_updated", "correction")):
            p.error("Preparing a proof requires parent, parity and correction evidence")
        prepare(a)
    elif a.action == "adopt":
        adopt(a.run.resolve())
    else:
        finalize(a.run.resolve())


if __name__ == "__main__":
    main()
