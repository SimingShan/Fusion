#!/usr/bin/env python3
"""Evaluate a completed training run and stage a worthwhile frozen candidate."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import re
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.codabench_submit import write_json


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def retains_original_function(saved, original):
    same_norm = (saved["normalization"].keys() == original["normalization"].keys()
                 and all(np.array_equal(v, original["normalization"][k]) for k, v in saved["normalization"].items()))
    if not same_norm:
        return False
    if (saved["model"] == original["model"] and saved["model_kwargs"] == original["model_kwargs"]
            and saved["state_dict"].keys() == original["state_dict"].keys()):
        return all(torch.equal(v, original["state_dict"][k]) for k, v in saved["state_dict"].items())
    if (saved["model"] != "pod_derived_inputs" or original["model"] != "pod_deeponet"
            or {k: v for k, v in saved["model_kwargs"].items() if k != "physical_schema"} != original["model_kwargs"]):
        return False
    state = saved["state_dict"]
    if any(k not in state or not torch.equal(v, state[k]) for k, v in original["state_dict"].items()):
        return False
    for key in ("derived_stem.weight", "derived_coefficients.weight", "derived_scalars.weight"):
        if key not in state or torch.count_nonzero(state[key]):
            return False
    if any(not torch.isfinite(v).all() for k, v in state.items() if k.startswith("derived_inputs.")):
        return False
    if torch.any(state["derived_inputs.std"] <= 0):
        return False
    for buffer, key in (("input_mean", "x_mean"), ("input_std", "x_std")):
        if not np.array_equal(state[f"derived_inputs.{buffer}"].numpy(), original["normalization"][key]):
            return False
    width = original["model_kwargs"]["input_dim"]
    for key in ("pressure_columns", "coils_columns", "history_columns"):
        if torch.any(state[f"derived_inputs.{key}"] >= width) or torch.any(state[f"derived_inputs.{key}"] < -2):
            return False
    return True


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--training", type=Path, required=True)
    p.add_argument("--candidate", required=True)
    p.add_argument("--baseline", type=Path, default=Path("results/candidates/20260919-001"))
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args()
    if not re.fullmatch(r"[0-9]{8}-[0-9]{3}", a.candidate):
        p.error("Use a neutral numeric candidate identifier")
    destination = ROOT / "results/candidates" / a.candidate
    if (destination / "decision.json").exists():
        raise ValueError("This candidate is already frozen; do not overwrite its evidence or checkpoints")
    train = ROOT / a.training
    metadata = json.loads((train / "run.json").read_text())
    if metadata.get("status") != "completed" or metadata.get("audit_labels_read") is not False:
        raise ValueError("Only complete development training runs are eligible")
    result = ROOT / "results/evaluations" / a.candidate
    result.mkdir(parents=True, exist_ok=True)
    baseline = json.loads((ROOT / a.baseline / "decision.json").read_text())
    saved = torch.load(train / "best.pt", map_location="cpu", weights_only=False)
    original = torch.load(ROOT / baseline["checkpoints"][1]["frozen_file"], map_location="cpu", weights_only=False)
    same = retains_original_function(saved, original)
    if same:
        write_json(result / "selection.json", {"accepted": False,
            "parameters_unchanged": saved["model"] == original["model"], "prediction_function_unchanged": True,
            "reason": "Training retained the already evaluated prediction function; no new candidate exists.", "audit_labels_read": False})
        print("Training retained the existing candidate; no duplicate evaluation or submission.", flush=True)
        return

    def run(script, *arguments):
        subprocess.run([sys.executable, str(ROOT / script), *map(str, arguments)], cwd=ROOT, check=True)

    native = result / "native"
    native.mkdir(exist_ok=True)
    shutil.copy2(train / "best.pt", native / "best.pt")
    run("scripts/train_pod.py", "--data-dir", "data/advanced_history_val_native_v3", "--output", native,
        "--predict-only", "--split", "val", "--workers", a.workers)
    # The three weights are declared before inspecting either new evaluation.
    for label, data, primary, secondary, split in (
            ("native_pair", "data/advanced_val_native_v3", "results/optimization/native_validation_v3/scaled_li", native, "val"),
            ("legacy_pair", "data/advanced_v1", "results/optimization/pod_scaled_li_v2", train, "legacy_val")):
        run("scripts/optimization/tune_pair_blend.py", "--data-dir", data, "--primary", primary,
            "--secondary", secondary, "--output", result / label, "--primary-weights", .25, .5, .75,
            "--splits", split, "--workers", a.workers)
    trials = []
    for weight in (.25, .5, .75):
        folder = f"gated_primary{round(weight*100):02d}"
        native_path = result / "native_pair" / folder / "val_score.json"
        legacy_path = result / "legacy_pair" / folder / "legacy_val_score.json"
        n, l = [json.loads(x.read_text())["S"] for x in (native_path, legacy_path)]
        trials.append({"weight": weight, "native": n, "legacy": l, "native_path": str(native_path), "legacy_path": str(legacy_path)})
    qualified = [x for x in trials if x["native"] >= baseline["validation"]["native"]["S"] + .0001
                 and x["legacy"] >= baseline["validation"]["legacy"]["S"] - .0001]
    write_json(result / "selection.json", {"trials": trials, "accepted": bool(qualified), "audit_labels_read": False})
    if not qualified:
        print("No new candidate passed both development thresholds.", flush=True)
        return
    selected = max(qualified, key=lambda x: x["native"])
    (destination / "checkpoints").mkdir(parents=True, exist_ok=True)
    records = []
    for name, path in (("a.pt", ROOT / baseline["checkpoints"][0]["frozen_file"]), ("b.pt", native / "best.pt")):
        frozen = destination / "checkpoints" / name
        shutil.copy2(path, frozen)
        records.append({"frozen_file": str(frozen), "sha256": digest(frozen)})
    paths = {"native": selected["native_path"], "legacy": selected["legacy_path"],
             "baseline_native": baseline["validation"]["native"]["path"], "baseline_legacy": baseline["validation"]["legacy"]["path"]}
    decision = {"candidate": a.candidate, "public_name": "Submission " + a.candidate,
        "frozen_utc": datetime.now(timezone.utc).isoformat(), "checkpoints": records,
        "primary_weight": selected["weight"], "gate_threshold": .1, "validation_tuned_weight": True,
        "test_labels_read": False, "audit_labels_read": False, "decision": "accept",
        "rationale": "Improved full-native development score with legacy robustness; three declared global weights compared.",
        "validation": {k: {"path": path, "sha256": digest(ROOT / path), "S": json.loads((ROOT / path).read_text())["S"]}
                       for k, path in paths.items()}}
    write_json(destination / "decision.json", decision)
    run("scripts/check_export_parity.py", "--candidate", destination)
    run("scripts/optimization/export_gated_ensemble.py", "--primary", destination / "checkpoints/a.pt",
        "--secondary", destination / "checkpoints/b.pt", "--primary-weight", selected["weight"],
        "--validation-tuned-weight", "--output", destination, "--device", "cpu", "--workers", a.workers)
    run("scripts/package_submission.py", "--pred", destination / "diii_d_public_test.npz", "--output", destination / "submission.zip")
    print("Validated candidate staged for the periodic submission process.", flush=True)


if __name__ == "__main__":
    main()
