#!/usr/bin/env python3
"""Check real parent sequences and reject unsafe changes to recovery reuse."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.challenge2.reuse_restart_export import compatible_plans, validate_sequence
from scripts.challenge2.check_fresh_initialization import prove_unchanged
from scripts.challenge2.reuse_valid_export import digest, read
from scripts.codabench_submit import write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise ValueError("Preserve previous checks")
    parent, run = [ROOT / f"results/challenge2/c2_export_{n:03}" for n in (45, 61)]
    compatible_plans(parent, run)
    checked, rejected = [], []

    def rejects(name, operation):
        try:
            operation()
        except (ValueError, AssertionError):
            rejected.append(name)
        else:
            raise AssertionError(f"Unsafe reuse accepted: {name}")

    for number in (44, 46):
        directory = ROOT / f"results/challenge2/c2_export_smoke_{number:03}"
        plan, record = read(directory / "export_plan.json"), read(directory / "shots/0000.json")
        validate_sequence(directory, plan, 0, record, digest(directory / "export_plan.json"))
        checked.append({"run": str(directory.relative_to(ROOT)), "frames": record["frames"],
                        "pressure_retry_frames": record["observed_pressure_retry_frames"]})
    assert checked[1]["pressure_retry_frames"] > 0
    directory = ROOT / "results/challenge2/c2_export_smoke_046"
    plan, original = read(directory / "export_plan.json"), read(directory / "shots/0000.json")
    for name, mutate in [
        ("failed_status", lambda x: x.update(status="failed")),
        ("missing_native_frame", lambda x: x["records"].pop()),
        ("runtime_cutoff", lambda x: x.update(runtime_limited_frames=1)),
        ("invalid_frame_scalar", lambda x: x["records"][0].update(scalar_valid=False)),
        ("inconsistent_scalar", lambda x: x["records"][0].update(q95=-1.)),
        ("wrong_native_time", lambda x: x["records"][0].update(time_ms=-1.)),
        ("wrong_plan", lambda x: x.update(plan_sha256="0"*64)),
        ("unrecorded_pressure_retry", lambda x: x.update(observed_pressure_retry_frames=0)),
        ("undeclared_recovery", lambda x: x["records"][0].update(initialization_retry_used=True)),
    ]:
        record = copy.deepcopy(original)
        mutate(record)
        rejects(name, lambda: validate_sequence(directory, plan, 0, record, digest(directory / "export_plan.json")))
    with tempfile.TemporaryDirectory(prefix="c2_reuse_", dir=ROOT / "results/checks") as tmp:
        source = Path(tmp) / "source"
        names = ("fusion/mast_physics.py", "scripts/challenge2/evaluate_profile_prior.py", "scripts/challenge2/export_mast_predictions.py")
        for name in names:
            destination = source / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(run / "source_snapshot" / name, destination)
        changes = [
            (names[0], "plasma = solver.solve(current)", "plasma = 2*solver.solve(current)", "changed_numerics"),
            (names[1], "import joblib", "import joblib\nimport random", "changed_imports"),
            (names[1], "if valid_prediction(original) or previous is None:", "if previous is None:", "changed_valid_return"),
            (names[2], 'info["nonlinear_profile_relative_residual"]<.02', 'info["nonlinear_profile_relative_residual"]<.03', "changed_sequence_state"),
        ]
        for name, old, new, label in changes:
            path = source / name
            saved = path.read_text()
            assert old in saved
            path.write_text(saved.replace(old, new))
            rejects(label, lambda: prove_unchanged(parent / "source_snapshot", source))
            path.write_text(saved)
        altered = Path(tmp) / "altered"
        (altered / "shots").mkdir(parents=True)
        with np.load(directory / "shots/0000.npz", allow_pickle=False) as arrays:
            values = {k: arrays[k].copy() for k in arrays.files}
        values["q95"][0] *= 2
        np.savez_compressed(altered / "shots/0000.npz", **values)
        record = copy.deepcopy(original)
        record["prediction_sha256"] = digest(altered / "shots/0000.npz")
        rejects("altered_payload_with_updated_hash", lambda: validate_sequence(altered, plan, 0, record, digest(directory / "export_plan.json")))
    result = {"passed": True, "valid_real_sequences": checked, "negative_checks": rejected,
              "negative_check_count": len(rejected), "test_labels_read": False, "audit_labels_read": False,
              "source_sha256": {name: digest(ROOT / name) for name in (
                  "scripts/challenge2/reuse_restart_export.py", "scripts/challenge2/check_fresh_initialization.py",
                  "scripts/challenge2/check_positive_current_trials.py", "scripts/challenge2/reuse_valid_export.py",
                  str(Path(__file__).relative_to(ROOT)))}, "eligible_for_submission": False}
    write_json(a.output, result)
    print({k: v for k, v in result.items() if k != "source_sha256"})


if __name__ == "__main__":
    main()
