#!/usr/bin/env python3
"""Prove and record reuse of complete valid sequences by a conditional retry.

The new predictor returns the original prediction when it is valid. Reuse is
therefore justified by induction over the identical previous-current updates,
but only for whole sequences whose original predictions are all valid. Failed
or incomplete sequences must be predicted again from their first frame.
"""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import ast
import copy
import fcntl
import hashlib
import importlib.metadata
import json
import os
import shutil

import joblib
import numpy as np


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def dump(node):
    return ast.dump(node, include_attributes=False)


def expr(text):
    return dump(ast.parse(text, mode="eval").body)


def function(tree, name):
    values = [x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == name]
    if len(values) != 1:
        raise ValueError(f"Expected one function: {name}")
    return copy.deepcopy(values[0])


class DefaultInference(ast.NodeTransformer):
    def __init__(self):
        self.branches, self.metadata = [], []

    def visit_If(self, node):
        if dump(node.test) in (expr('"pressure_factor_model" in prior'), expr("observed_pressure_only")):
            if node.orelse:
                raise ValueError("Unexpected alternative branch")
            self.branches.append(dump(node.test))
            return None
        return self.generic_visit(node)

    def visit_Call(self, node):
        expected = {"pressure_factor_from_prior": expr('measured_pressure and "pressure_factor_model" in prior'),
                    "observed_pressure_only": expr("bool(observed_pressure_only)")}
        for keyword in list(node.keywords):
            if keyword.arg in expected:
                if dump(node.func) != expr("details.update") or dump(keyword.value) != expected[keyword.arg]:
                    raise ValueError("Changed metadata affects the computation")
                self.metadata.append(keyword.arg)
                node.keywords.remove(keyword)
        return self.generic_visit(node)


class ExportMetadata(ast.NodeTransformer):
    """Remove only the new receipt fields, never prediction computations."""
    fields = {"observed_pressure_retry_frames", "pressure_observation_retry_used", "original_prediction_failure"}

    def visit_Tuple(self, node):
        node.elts = [x for x in node.elts if not (isinstance(x, ast.Constant) and x.value in self.fields)]
        return self.generic_visit(node)

    def visit_Dict(self, node):
        pairs = [(k, v) for k, v in zip(node.keys, node.values)
                 if not (isinstance(k, ast.Constant) and k.value == "observed_pressure_retry_frames")]
        node.keys, node.values = [x[0] for x in pairs], [x[1] for x in pairs]
        return self.generic_visit(node)


def prove_default_branch(old_source, new_source, prior):
    if "pressure_factor_model" in prior:
        raise ValueError("The default prior must not contain the optional factor estimator")
    trees = [ast.parse((p / "scripts/challenge2/evaluate_profile_prior.py").read_text())
             for p in (old_source, new_source)]
    # Module setup and imports must be identical; all imported local numerical
    # dependencies are separately required to have identical file hashes.
    prefixes = []
    for tree in trees:
        first = next(i for i, x in enumerate(tree.body) if isinstance(x, ast.FunctionDef))
        prefixes.append(dump(ast.Module(body=tree.body[:first], type_ignores=[])))
    if prefixes[0] != prefixes[1]:
        raise ValueError("Inference module setup changed")
    old, new = [function(tree, "infer") for tree in trees]
    if new.args.args[-1].arg != "observed_pressure_only" or dump(new.args.defaults[-1]) != expr("False"):
        raise ValueError("The new default branch is not disabled")
    new.args.args.pop(); new.args.defaults.pop()
    normalizer = DefaultInference()
    new = normalizer.visit(new)
    if (sorted(normalizer.branches) != sorted([expr('"pressure_factor_model" in prior'), expr("observed_pressure_only")])
            or sorted(normalizer.metadata) != ["observed_pressure_only", "pressure_factor_from_prior"]
            or dump(old) != dump(new)):
        raise ValueError("The original default numerical function changed")
    expected_valid = ast.parse('''
def valid_prediction(result):
    field, _, info = result
    return (bool(info.get("scalar_valid")) and np.isfinite(field).all()
            and all(info.get(key) is not None and np.isfinite(info[key]) and info[key] > 0
                    for key in ("q95", "betaN")))
''').body[0]
    if dump(function(trees[1], "valid_prediction")) != dump(expected_valid):
        raise ValueError("The conditional validity rule changed")
    retry = function(trees[1], "infer_with_retry")
    original_attempt = next(x for x in retry.body if isinstance(x, ast.Try))
    expected_prefix = ast.parse('''
original = infer(row, frame, prior, solver, envelope, previous, max_seconds, max_evaluations,
                 measured_pressure, pressure_factor, False)
if valid_prediction(original):
    original[2]["pressure_observation_retry_used"] = False
    return original
''').body
    if [dump(x) for x in original_attempt.body[:2]] != [dump(x) for x in expected_prefix]:
        raise ValueError("Valid original outputs are not returned unchanged")
    exporters = [ast.parse((p / "scripts/challenge2/export_mast_predictions.py").read_text())
                 for p in (old_source, new_source)]
    old_init, new_init = [function(tree, "initialize") for tree in exporters]
    found = 0
    for node in list(new_init.body):
        if isinstance(node, ast.ImportFrom) and node.module == "scripts.challenge2.evaluate_profile_prior":
            if [x.name for x in node.names] != ["infer", "infer_with_retry"]:
                raise ValueError("Unexpected prediction imports")
            node.names.pop()
        if isinstance(node, ast.If) and dump(node.test) == expr('plan.get("retry_observed_pressure")'):
            if node.orelse or [dump(x) for x in node.body] != [dump(ast.parse("infer=infer_with_retry").body[0])]:
                raise ValueError("Unexpected predictor dispatch")
            new_init.body.remove(node); found += 1
    if found != 1 or dump(old_init) != dump(new_init):
        raise ValueError("Export initialization changed")
    old_shot, new_shot = [function(tree, "predict_shot") for tree in exporters]
    if dump(old_shot) != dump(ExportMetadata().visit(new_shot)):
        raise ValueError("Native sequence or previous-current updates changed")
    return {"default_inference_ast_equal": True, "native_sequence_ast_equal": True,
            "valid_prediction_returned_unchanged": True, "optional_factor_estimator_absent": True}


def checked_plan(run):
    plan = read(run / "export_plan.json")
    if plan.get("test_labels_read") is not False or plan.get("audit_labels_read") is not False:
        raise ValueError("Only input-only exports can be reused")
    if digest(Path(plan["raw_root"]) / "download_manifest.json") != plan["inventory_sha256"]:
        raise ValueError("Pinned input inventory changed")
    if digest(run / "checkpoints/prior.joblib") != plan["prior_sha256"]:
        raise ValueError("Pinned prior changed")
    for name, expected in plan["source_sha256"].items():
        if digest(run / "source_snapshot" / name) != expected:
            raise ValueError("Frozen source changed")
    for name, expected in plan["packages"].items():
        if importlib.metadata.version(name) != expected:
            raise ValueError("Scientific dependency version changed")
    return plan


def checked_smoke(run):
    plan, report = checked_plan(run), read(run / "mast_export.json")
    if (report["status"] != "smoke_complete" or report["scope"] != "smoke"
            or report["runtime_limited_frames"] or report["frames"] != report["scalar_valid"]
            or report["plan_sha256"] != digest(run / "export_plan.json")
            or report["source_sha256"] != plan["source_sha256"]
            or report["prediction_sha256"] != digest(run / "mast_public_test.npz")):
        raise ValueError("A smoke check is incomplete or changed")
    return plan, report


def prepare(args):
    parent, run = args.parent.resolve(), args.run.resolve()
    target = run / "reuse_proof.json"
    if target.exists():
        raise ValueError("Reuse proof is already frozen")
    old, new = checked_plan(parent), checked_plan(run)
    ignored = {"created_utc", "source_sha256", "retry_observed_pressure"}
    if {k: v for k, v in old.items() if k not in ignored} != {k: v for k, v in new.items() if k not in ignored}:
        raise ValueError("Prediction plans differ beyond the conditional retry")
    if old.get("retry_observed_pressure", False) or new.get("retry_observed_pressure") is not True:
        raise ValueError("Expected an original default and a conditional retry")
    unused = {"fusion/advanced_models.py", "fusion/geometry_loss.py", "fusion/input_derived.py"}
    changed = {k for k in set(old["source_sha256"]) | set(new["source_sha256"])
               if old["source_sha256"].get(k) != new["source_sha256"].get(k)}
    if changed - unused - {"scripts/challenge2/evaluate_profile_prior.py", "scripts/challenge2/export_mast_predictions.py"}:
        raise ValueError("A numerical dependency changed")
    # These three modules belong only to the separate DIII-D predictor. Any
    # future import from the checked numerical path must invalidate the proof.
    for name in set(old["source_sha256"]) - unused:
        if name.endswith(".py"):
            for root in (parent, run):
                tree = ast.parse((root / "source_snapshot" / name).read_text())
                for node in ast.walk(tree):
                    modules = ([node.module or ""] if isinstance(node, ast.ImportFrom) else
                               [x.name for x in node.names] if isinstance(node, ast.Import) else [])
                    if any(m in ("fusion.advanced_models", "fusion.geometry_loss", "fusion.input_derived") for m in modules):
                        # An unrelated DIII-D module in the broad snapshot is
                        # not on the MAST import path; enumerate it explicitly.
                        if name not in {"fusion/geometry_targets.py"}:
                            raise ValueError(f"Unexpected dependency on a changed DIII-D module: {name}")
    checks = prove_default_branch(parent / "source_snapshot", run / "source_snapshot",
                                  joblib.load(run / "checkpoints/prior.joblib"))
    before, after, correction = args.parity_original.resolve(), args.parity_updated.resolve(), args.correction.resolve()
    old_smoke, old_report = checked_smoke(before)
    new_smoke, new_report = checked_smoke(after)
    fixed_smoke, fixed_report = checked_smoke(correction)
    if (old_smoke["source_sha256"] != old["source_sha256"]
            or new_smoke["source_sha256"] != new["source_sha256"]
            or fixed_smoke["source_sha256"] != new["source_sha256"]):
        raise ValueError("Smoke checks did not execute these exact frozen sources")
    for smoke in (old_smoke, new_smoke, fixed_smoke):
        for key in ("prior_sha256", "packages", "inventory_sha256", "max_evaluations", "max_seconds", "pressure_factor", "allow_approximate_seed"):
            if smoke[key] != new[key]:
                raise ValueError("Smoke check settings differ")
    if (old_smoke["shots"] != new_smoke["shots"] or old_report["prediction_sha256"] != new_report["prediction_sha256"]
            or new_report["observed_pressure_retry_frames"] != 0 or fixed_report["observed_pressure_retry_frames"] < 1):
        raise ValueError("Require byte-identical unchanged output and a completed corrected sequence")
    runner = run / "reuse" / Path(__file__).name
    runner.parent.mkdir(exist_ok=True)
    shutil.copyfile(__file__, runner)
    evidence = [{"path": str(p), "sha256": digest(p)} for root in (before, after, correction)
                for p in (root / "export_plan.json", root / "mast_export.json")]
    proof = {"created_utc": datetime.now(timezone.utc).isoformat(), "passed": True,
             "parent": str(parent), "run": str(run), "parent_plan_sha256": digest(parent / "export_plan.json"),
             "new_plan_sha256": digest(run / "export_plan.json"), "runner": str(runner), "runner_sha256": digest(runner),
             "checks": checks, "evidence": evidence, "parity_frames": new_report["frames"],
             "corrected_frames": fixed_report["frames"], "test_labels_read": False, "audit_labels_read": False,
             "rule": "Reuse only complete scalar-valid finite sequences without a runtime cutoff; recompute every other sequence from frame zero"}
    write(target, proof)
    print(json.dumps({"passed": True, "parity_frames": proof["parity_frames"], "corrected_frames": proof["corrected_frames"]}))


def validate_shot(parent, plan, index, record, plan_hash):
    import pyarrow.parquet as pq
    item = plan["shots"][index]
    source = parent / "shots" / f"{index:04d}.npz"
    if (record["status"] != "complete" or record["index"] != index or record["raw_path"] != item["path"]
            or record["plan_sha256"] != plan_hash or record["scalar_valid"] != record["frames"]
            or record["runtime_limited_frames"] or record.get("observed_pressure_retry_frames", 0)
            or record["prediction_sha256"] != digest(source)):
        raise ValueError("A parent sequence is not eligible for reuse")
    raw = Path(plan["raw_root"]) / item["path"]
    if raw.stat().st_size != item["size"]:
        raise ValueError("Pinned native input changed")
    times = np.asarray(pq.read_table(raw, columns=["efit_times"])["efit_times"][0].as_py(), dtype="<f8")
    n = len(times)
    if (n != record["frames"] or not np.isfinite(times).all() or np.any(np.diff(times) <= 0)
            or hashlib.sha256(times.tobytes()).hexdigest() != record["native_times_sha256"]
            or len(record["records"]) != n):
        raise ValueError("Native time coverage changed")
    frames = record["records"]
    if any(x["frame"] != i or x["time_ms"] != times[i] or not x["scalar_valid"]
           or x["solver_elapsed_seconds"] >= plan["max_seconds"] for i, x in enumerate(frames)):
        raise ValueError("A frame is invalid or depends on the runtime cutoff")
    with np.load(source, allow_pickle=False) as arrays:
        if set(arrays.files) != {"psirz", "q95", "betaN"}:
            raise ValueError("Unexpected prediction contents")
        for key, shape in (("psirz", (n, 65, 65)), ("q95", (n,)), ("betaN", (n,))):
            value = arrays[key]
            if value.shape != shape or value.dtype != np.float32 or not np.isfinite(value).all():
                raise ValueError("Invalid prediction array")
            if key != "psirz" and (np.any(value <= 0) or not np.array_equal(value, np.asarray([x[key] for x in frames], np.float32))):
                raise ValueError("Invalid or inconsistent scalar array")


def adopt(run, indices=None):
    proof_path = run / "reuse_proof.json"
    proof, proof_hash = read(proof_path), digest(proof_path)
    if not proof["passed"] or Path(__file__).resolve() != Path(proof["runner"]).resolve() or digest(__file__) != proof["runner_sha256"]:
        raise ValueError("Execute the frozen, checked reuse runner")
    parent = Path(proof["parent"])
    if digest(parent / "export_plan.json") != proof["parent_plan_sha256"] or digest(run / "export_plan.json") != proof["new_plan_sha256"]:
        raise ValueError("A prediction plan changed")
    for item in proof["evidence"]:
        if digest(item["path"]) != item["sha256"]:
            raise ValueError("Checked evidence changed")
    old, new = checked_plan(parent), checked_plan(run)
    selected = list(range(len(old["shots"]))) if indices is None else indices
    if len(set(selected)) != len(selected) or any(i < 0 or i >= len(old["shots"]) for i in selected):
        raise ValueError("Invalid reuse indices")
    reused, skipped = [], []
    with (run / "reuse/adopt.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for index in selected:
            original = parent / "shots" / f"{index:04d}.json"
            target = run / "shots" / f"{index:04d}.json"
            if target.exists():
                existing = read(target)
                if existing.get("status") == "complete":
                    if existing["plan_sha256"] != proof["new_plan_sha256"] or digest(target.with_suffix(".npz")) != existing["prediction_sha256"]:
                        raise ValueError("A completed child output changed")
                    continue
            if not original.exists() or read(original).get("status") != "complete":
                skipped.append(index)
                continue
            record, original_hash = read(original), digest(original)
            if record["runtime_limited_frames"]:
                skipped.append(index)
                continue
            validate_shot(parent, old, index, record, proof["parent_plan_sha256"])
            target.parent.mkdir(exist_ok=True)
            prediction = target.with_suffix(".npz")
            temporary = prediction.with_name(prediction.name + f".{os.getpid()}.tmp")
            shutil.copyfile(original.with_suffix(".npz"), temporary)
            if digest(temporary) != record["prediction_sha256"] or digest(original) != original_hash:
                raise ValueError("Parent output changed while copying")
            temporary.replace(prediction)
            record.update(plan_sha256=proof["new_plan_sha256"], observed_pressure_retry_frames=0,
                          reuse={"type": "unchanged_valid_default_sequence", "proof_sha256": proof_hash,
                                 "parent_plan_sha256": proof["parent_plan_sha256"], "parent_receipt": str(original),
                                 "parent_receipt_sha256": original_hash, "prediction_recomputed": False})
            for frame in record["records"]:
                frame.update(pressure_observation_retry_used=False, original_prediction_failure=None)
            write(target, record)
            reused.append({"index": index, "frames": record["frames"], "prediction_sha256": record["prediction_sha256"]})
        report = {"created_utc": datetime.now(timezone.utc).isoformat(), "proof_sha256": proof_hash,
                  "reused_this_pass": reused, "reused_shots": len(reused), "reused_frames": sum(x["frames"] for x in reused),
                  "indices_requiring_prediction": skipped, "test_labels_read": False, "audit_labels_read": False}
        write(run / "reuse" / f"adoption_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')}.json", report)
    print(json.dumps({k: v for k, v in report.items() if k not in ("reused_this_pass", "indices_requiring_prediction")}
                     | {"remaining_sequences": len(skipped)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--parity-original", type=Path)
    parser.add_argument("--parity-updated", type=Path)
    parser.add_argument("--correction", type=Path)
    parser.add_argument("--indices", type=int, nargs="*")
    args = parser.parse_args()
    if args.prepare:
        if any(getattr(args, key) is None for key in ("parent", "parity_original", "parity_updated", "correction")):
            parser.error("Preparation requires the parent and all three completed checks")
        prepare(args)
    else:
        adopt(args.run.resolve(), args.indices)


if __name__ == "__main__":
    main()
