#!/usr/bin/env python3
"""Fixed primary/current quality gate and fixed 75/25 blend on valid inputs."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.optimization.quality_gate import feature_columns, gate_from_inputs, write_array, evaluate, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/advanced_v1"))
    parser.add_argument("--test-dir", type=Path, default=Path("data/advanced_test_v1"))
    parser.add_argument("--primary", "--fourier", dest="fourier", type=Path, default=Path("results/optimization/pod_fourier_full_v1"))
    parser.add_argument("--expert-name", default="Fourier POD DeepONet")
    parser.add_argument("--secondary", "--currents", dest="currents", type=Path, default=Path("results/optimization/pod_currents64_cpu"))
    parser.add_argument("--secondary-name", default="currents-only POD DeepONet")
    parser.add_argument("--output", type=Path, default=Path("results/optimization/quality_gate_fourier_v2"))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--splits", nargs="+", choices=("val", "legacy_val"), default=["val", "legacy_val"])
    parser.add_argument("--skip-test-coverage", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.data_dir / "manifest.json").read_text())
    columns = feature_columns(manifest["feature_names"])
    sources = {split: {"fourier": args.fourier / f"{split}_predictions.npz",
                        "currents": args.currents / f"{split}_predictions.npz"} for split in args.splits}
    policy = {"created_utc": datetime.now(timezone.utc).isoformat(), "gate": f"use {args.secondary_name} if at most 10% of real edge channels have positive observed Te and ne",
              "primary_expert": args.expert_name, "primary_directory": str(args.fourier),
              "secondary_expert": args.secondary_name, "secondary_directory": str(args.currents),
              "primary_gate": f"otherwise {args.expert_name}", "gated_blend75": f"otherwise 0.75 {args.expert_name} + 0.25 {args.secondary_name}",
              "threshold": 0.1, "fourier_weight": 0.75, "threshold_search": False, "weight_search": False,
              "audit_labels_read": False, "decisions_use_labels_or_shot_id": False,
              "script_sha256": sha256(Path(__file__)), "source_manifest_sha256": sha256(args.data_dir / "manifest.json"),
              "sources": {split: {name: {"path": str(path), "sha256": sha256(path)} for name, path in paths.items()} for split, paths in sources.items()}}
    policy_path = args.output / "policy.json"
    policy_path.write_text(json.dumps(policy, indent=2) + "\n")
    policy_hash = sha256(policy_path)
    summary = {"evaluation_type": "predeclared_gated_expert_blend", "primary_expert": args.expert_name,
               "primary_directory": str(args.fourier), "secondary_expert": args.secondary_name,
               "secondary_directory": str(args.currents), "policy_sha256": policy_hash,
               "audit_labels_read": False, "leaderboard_submission": False, "splits": {}}
    # Construct both folds before reading any labels through the official scorer.
    for split, paths in sources.items():
        entries = manifest[split]
        output_paths = {method: args.output / method / f"{split}_predictions.npz" for method in ("gate", "gated_blend75")}
        for path in output_paths.values():
            path.parent.mkdir(exist_ok=True)
        archives = {name: zipfile.ZipFile(path.with_suffix(".tmp.npz"), "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) for name, path in output_paths.items()}
        coverage = {"shots": len(entries), "frames": 0, "gated_frames": 0, "gated_shots": 0}
        try:
            with np.load(paths["fourier"], allow_pickle=False) as fourier, np.load(paths["currents"], allow_pickle=False) as currents:
                expected = {f"shot_{i:04d}_{name}" for i in range(len(entries)) for name in ("psirz", "q95", "betaN")}
                if set(fourier.files) != expected or set(currents.files) != expected:
                    raise ValueError("Expert prediction keys disagree with the complete fold")
                for i, entry in enumerate(entries):
                    with np.load(args.data_dir / entry["path"], allow_pickle=False) as shot:
                        X = shot["X"]
                    gated, _, _, _ = gate_from_inputs(X, columns)
                    coverage["frames"] += len(X)
                    coverage["gated_frames"] += int(gated.sum())
                    coverage["gated_shots"] += int(gated.any())
                    for field in ("psirz", "q95", "betaN"):
                        key = f"shot_{i:04d}_{field}"
                        A, B = fourier[key], currents[key]
                        expected_shape = (len(X), 65, 65) if field == "psirz" else (len(X),)
                        if A.shape != expected_shape or B.shape != expected_shape:
                            raise ValueError(f"Expert shape mismatch {key}")
                        mask = gated[:, None, None] if field == "psirz" else gated
                        write_array(archives["gate"], key, np.where(mask, B, A))
                        write_array(archives["gated_blend75"], key, np.where(mask, B, 0.75 * A + 0.25 * B))
        finally:
            for archive in archives.values():
                archive.close()
        for path in output_paths.values():
            os.replace(path.with_suffix(".tmp.npz"), path)
        summary["splits"][split] = {"coverage": coverage, "predictions": {name: str(path) for name, path in output_paths.items()}, "scores": {}}
    for split, item in summary["splits"].items():
        for method, path in item["predictions"].items():
            score_path = args.output / method / f"{split}_score.json"
            score = evaluate(args.data_dir, split, Path(path), score_path, args.workers, policy_hash, method)
            item["scores"][method] = {key: value for key, value in score.items() if key not in ("evaluation", "experiment")}
            print(f"{split} {method}: S={score['S']:.6f}", flush=True)
    if not args.skip_test_coverage:
        test = json.loads((args.test_dir / "manifest.json").read_text())
        if test["feature_names"] != manifest["feature_names"]:
            raise ValueError("Public-test feature order differs from validation")
        coverage = {"shots": len(test["test"]), "frames": 0, "gated_frames": 0, "gated_shots": 0}
        for entry in test["test"]:
            with np.load(args.test_dir / entry["path"], allow_pickle=False) as shot:
                selected, _, _, _ = gate_from_inputs(shot["X"], columns)
            coverage["frames"] += len(selected)
            coverage["gated_frames"] += int(selected.sum())
            coverage["gated_shots"] += int(selected.any())
        summary["public_test_input_only_coverage"] = coverage
        print(json.dumps(coverage), flush=True)
    for split, paths in sources.items():
        for name, path in paths.items():
            if sha256(path) != policy["sources"][split][name]["sha256"]:
                raise ValueError("Source expert changed during scoring")
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
