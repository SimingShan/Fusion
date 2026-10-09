#!/usr/bin/env python3
"""Distribute a frozen export over independent, disjoint complete-shot jobs.

This changes scheduling only. All predictions and final validation are executed
by the original immutable exporter, against its original immutable plan.
"""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
import shutil
import sys


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def partition_indices(count, parts):
    if not 1 <= parts <= count:
        raise ValueError("Require between one and the number of shots partitions")
    return [list(range(i, count, parts)) for i in range(parts)]


def prepare(run, parts):
    path = run / "execution_plan.json"
    if path.exists():
        raise ValueError("Execution plan is already frozen")
    prediction = json.loads((run / "export_plan.json").read_text())
    indices = partition_indices(len(prediction["shots"]), parts)
    snapshot = run / "execution" / Path(__file__).name
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(__file__, snapshot)
    value = {"created_utc": datetime.now(timezone.utc).isoformat(),
             "export_plan_sha256": digest(run / "export_plan.json"),
             "runner": str(snapshot), "runner_sha256": digest(snapshot),
             "parts": indices, "scheduling_only": True,
             "prediction_plan_unchanged": True}
    write_json(path, value)
    print(json.dumps({"frozen": True, "shots": len(prediction["shots"]), "parts": parts}))


def load_exporter(run):
    source = run / "source_snapshot/scripts/challenge2/export_mast_predictions.py"
    specification = importlib.util.spec_from_file_location("_frozen_mast_export", source)
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    module.initialize(run)
    snapshot = (run / "source_snapshot").resolve()
    for name in ("scripts.challenge2.evaluate_profile_prior", "fusion.mast_profiles",
                 "fusion.mast_physics", "fusion.profile_scalars"):
        if not Path(sys.modules[name].__file__).resolve().is_relative_to(snapshot):
            raise ValueError("Prediction module was loaded outside its frozen snapshot")
    return module


def check_execution(run):
    path = run / "execution_plan.json"
    plan = json.loads(path.read_text())
    if digest(run / "export_plan.json") != plan["export_plan_sha256"]:
        raise ValueError("Frozen prediction plan changed")
    if Path(__file__).resolve() != Path(plan["runner"]).resolve() or digest(__file__) != plan["runner_sha256"]:
        raise ValueError("Execute the frozen scheduling runner")
    count = len(json.loads((run / "export_plan.json").read_text())["shots"])
    if plan["parts"] != partition_indices(count, len(plan["parts"])):
        raise ValueError("Shot partitions are not complete and disjoint")
    return plan, digest(path)


def run_part(run, execution, execution_hash, part):
    if not 0 <= part < len(execution["parts"]):
        raise ValueError("Invalid partition")
    directory = run / "execution"
    with (directory / f"part_{part:02d}.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        exporter = load_exporter(run)
        summaries, failures = [], []
        assigned = execution["parts"][part]
        for index in assigned:
            try:
                summaries.append(exporter.predict_shot(index))
            except Exception as exc:
                failures.append({"index": index, "error": f"{type(exc).__name__}: {exc}"})
            record = {"part": part, "execution_plan_sha256": execution_hash,
                      "assigned_indices": assigned, "summaries": summaries, "failures": failures,
                      "status": "running", "completed_shots": len(summaries),
                      "completed_frames": sum(x["frames"] for x in summaries)}
            write_json(directory / f"part_{part:02d}.json", record)
            print(json.dumps({"part": part, "completed_shots": len(summaries),
                              "failed_shots": len(failures), "assigned_shots": len(assigned)}), flush=True)
        record["status"] = "failed" if failures else "complete"
        write_json(directory / f"part_{part:02d}.json", record)
        if failures:
            raise RuntimeError("Partition has failed shots; retain successful outputs for diagnosis")


def assemble(run, execution, execution_hash):
    with (run / "execution/assemble.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (run / "mast_export.json").exists():
            raise ValueError("A completed export already exists; preserve it")
        exporter = load_exporter(run)
        summaries = []
        for part, indices in enumerate(execution["parts"]):
            record = json.loads((run / "execution" / f"part_{part:02d}.json").read_text())
            if (record["status"] != "complete" or record["failures"]
                    or record["execution_plan_sha256"] != execution_hash
                    or record["assigned_indices"] != indices
                    or [x["index"] for x in record["summaries"]] != indices):
                raise ValueError("Incomplete or inconsistent execution partition")
            for index in indices:
                # Reuse the original exporter's independent plan/payload hash check.
                verified = exporter.predict_shot(index)
                recorded = record["summaries"][indices.index(index)]
                if verified != recorded:
                    raise ValueError("Stored shot summary changed")
                summaries.append(verified)
        summaries.sort(key=lambda x: x["index"])
        exporter.assemble(run, exporter.WORK["plan"], summaries)
        path = run / "mast_export.json"
        report = json.loads(path.read_text())
        report["execution_plan_sha256"] = execution_hash
        write_json(path, report)
        write_json(run / "progress.json", {"completed_shots": len(summaries), "failures": [],
                   "completed_frames": sum(x["frames"] for x in summaries),
                   "total_shots": len(summaries), "converged_frames": sum(x["converged"] for x in summaries),
                   "summaries": summaries})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare", action="store_true")
    action.add_argument("--check-only", action="store_true")
    action.add_argument("--part", type=int)
    action.add_argument("--assemble", action="store_true")
    parser.add_argument("--parts", type=int, default=32)
    args = parser.parse_args()
    run = args.run.resolve()
    if args.prepare:
        prepare(run, args.parts)
    elif args.check_only:
        exporter = load_exporter(run)
        count = len(exporter.WORK["plan"]["shots"])
        partition = partition_indices(count, min(count, args.parts))
        assert sorted(i for group in partition for i in group) == list(range(count))
        print(json.dumps({"passed": True, "frozen_imports": True, "disjoint_complete_partitions": True,
                          "shots": count, "parts": len(partition)}))
    else:
        execution, execution_hash = check_execution(run)
        if args.assemble:
            assemble(run, execution, execution_hash)
        else:
            run_part(run, execution, execution_hash, args.part)


if __name__ == "__main__":
    main()
