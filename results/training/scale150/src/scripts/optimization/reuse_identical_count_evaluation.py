#!/usr/bin/env python3
"""Reuse existing evaluation only after exact frozen predictor equality."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import numpy as np
import torch


def identical(first, second):
    for key in ("model", "model_kwargs"):
        if first[key] != second[key]:
            return False
    if first["run"]["feature_names"] != second["run"]["feature_names"]:
        return False
    for key in ("state_dict", "normalization"):
        if first[key].keys() != second[key].keys():
            return False
        for name in first[key]:
            a, b = first[key][name], second[key][name]
            if isinstance(a, torch.Tensor):
                if not torch.equal(a, b):
                    return False
            elif not np.array_equal(a, b):
                return False
    return True


def link_tree(source, destination):
    for path in source.rglob("*"):
        if path.is_file():
            target = destination / path.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                raise ValueError(f"Refusing to replace existing evaluation artifact: {target}")
            try:
                os.link(path, target)
            except OSError:
                shutil.copy2(path, target)


def main():
    source_path = Path("results/optimization/pod_history128_robust_seed2027_v4/best.pt")
    source = torch.load(source_path, map_location="cpu", weights_only=False)
    sources = {"capped": Path("results/optimization/quality_gate_scaled_history128_v7"),
               "native": Path("results/optimization/native_history128_v7/robust_history128"),
               "native_combined": Path("results/optimization/native_history128_v7/combined")}
    for path in (sources["capped"] / "summary.json", sources["native"] / "val_score.json", sources["native_combined"] / "summary.json"):
        if not path.exists():
            print("Source evaluation not complete; running normal evaluation", flush=True)
            return
    for arm in ("control", "weighted"):
        checkpoint = Path(f"results/optimization/pod_history128_count_{arm}_v5/best.pt")
        candidate = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if not identical(source, candidate):
            continue
        destination = Path(f"results/optimization/native_count_ablation_v5/{arm}")
        for name, directory in sources.items():
            link_tree(directory, destination / name)
        report = {"reason": "Frozen predictor exactly equals already evaluated baseline",
                  "model_kwargs_equal": True, "feature_contract_equal": True,
                  "all_state_tensors_bitwise_equal": True, "all_normalization_arrays_equal": True,
                  "candidate_checkpoint": str(checkpoint.resolve()),
                  "candidate_checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                  "baseline_checkpoint": str(source_path.resolve()),
                  "baseline_checkpoint_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
                  "evaluations_reused": {k: str(v.resolve()) for k, v in sources.items()},
                  "new_label_reads": False, "audit_labels_read": False}
        (destination / "reused_identical_model.json").write_text(json.dumps(report, indent=2)+"\n")
        print(f"Reused {arm} evaluation after exact frozen-model equality", flush=True)


if __name__ == "__main__":
    main()
