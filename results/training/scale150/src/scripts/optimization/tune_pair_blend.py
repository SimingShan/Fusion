#!/usr/bin/env python3
"""Explicit validation selection among global pair weights with a fixed gate."""
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
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--secondary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--primary-weights", type=float, nargs="+", default=[0.25, 0.5])
    parser.add_argument("--existing-075-summary", type=Path)
    parser.add_argument("--splits", nargs="+", choices=["val", "legacy_val"], default=["val", "legacy_val"])
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if any(not 0 < value < 1 for value in args.primary_weights) or len(set(args.primary_weights)) != len(args.primary_weights):
        parser.error("Weights must be distinct fractions strictly between zero and one")
    methods = {f"gated_primary{round(weight*100):02d}": weight for weight in args.primary_weights}
    if len(methods) != len(args.primary_weights):
        parser.error("Weight names must be distinct at integer-percent precision")
    manifest = json.loads((args.data_dir / "manifest.json").read_text())
    columns = feature_columns(manifest["feature_names"])
    sources = {split: {"primary": args.primary / f"{split}_predictions.npz",
                       "secondary": args.secondary / f"{split}_predictions.npz"} for split in args.splits}
    args.output.mkdir(parents=True, exist_ok=True)
    policy = {
        "evaluation_type": "validation_tuned_global_pair_weights", "created_utc": datetime.now(timezone.utc).isoformat(),
        "primary_expert": "scaled-li POD DeepONet", "secondary_expert": "robust history128 li POD DeepONet",
        "candidate_primary_weights": [0.25, 0.5, 0.75], "new_weights_evaluated": args.primary_weights,
        "existing_075_summary": str(args.existing_075_summary) if args.existing_075_summary else None,
        "weight_selection_uses_validation_labels": True, "global_weights_only": True,
        "selection_data": "same fresh validation shot identities; legacy fold provides a robustness comparison",
        "gate": "secondary alone when at most 10% of real edge channels have positive observed Te and ne",
        "gate_threshold": 0.1, "gate_threshold_tuned": False, "gate_uses_labels_or_shot_identity": False,
        "test_labels_read": False, "audit_labels_read": False, "official_submission": False,
        "source_manifest_sha256": sha256(args.data_dir / "manifest.json"), "script_sha256": sha256(Path(__file__)),
        "sources": {split: {name: {"path": str(path.resolve()), "sha256": sha256(path)} for name, path in paths.items()}
                    for split, paths in sources.items()},
    }
    (args.output / "policy.json").write_text(json.dumps(policy, indent=2)+"\n")
    policy_hash = sha256(args.output / "policy.json")
    summary = {"evaluation_type": policy["evaluation_type"], "policy_sha256": policy_hash,
               "audit_labels_read": False, "leaderboard_submission": False, "splits": {}}
    # Freeze all declared mixtures on all requested folds before official scoring.
    for split, source in sources.items():
        entries = manifest[split]
        paths = {name: args.output / name / f"{split}_predictions.npz" for name in methods}
        for path in paths.values():
            path.parent.mkdir(exist_ok=True)
        archives = {name: zipfile.ZipFile(path.with_suffix(".tmp.npz"), "w", compression=zipfile.ZIP_DEFLATED,
                                        compresslevel=1, allowZip64=True) for name, path in paths.items()}
        coverage = {"shots": len(entries), "frames": 0, "gated_frames": 0, "gated_shots": 0}
        try:
            with np.load(source["primary"], allow_pickle=False) as primary, np.load(source["secondary"], allow_pickle=False) as secondary:
                expected = {f"shot_{i:04d}_{field}" for i in range(len(entries)) for field in ("psirz", "q95", "betaN")}
                if set(primary.files) != expected or set(secondary.files) != expected:
                    raise ValueError("Expert keys differ from the complete selected fold")
                for i, entry in enumerate(entries):
                    with np.load(args.data_dir / entry["path"], allow_pickle=False) as shot:
                        X = shot["X"]
                    gated, _, _, _ = gate_from_inputs(X, columns)
                    coverage["frames"] += len(X)
                    coverage["gated_frames"] += int(gated.sum())
                    coverage["gated_shots"] += int(gated.any())
                    for field in ("psirz", "q95", "betaN"):
                        key = f"shot_{i:04d}_{field}"
                        A, B = primary[key], secondary[key]
                        expected_shape = (len(X), 65, 65) if field == "psirz" else (len(X),)
                        if A.shape != expected_shape or B.shape != expected_shape:
                            raise ValueError(f"Expert shape mismatch: {key}")
                        mask = gated[:, None, None] if field == "psirz" else gated
                        for method, weight in methods.items():
                            write_array(archives[method], key, np.where(mask, B, weight*A+(1-weight)*B))
        finally:
            for archive in archives.values():
                archive.close()
        for path in paths.values():
            os.replace(path.with_suffix(".tmp.npz"), path)
        summary["splits"][split] = {"coverage": coverage, "scores": {},
                                    "predictions": {key: str(value) for key, value in paths.items()}}
    for split, item in summary["splits"].items():
        for method, prediction in item["predictions"].items():
            score = evaluate(args.data_dir, split, Path(prediction), args.output / method / f"{split}_score.json",
                             args.workers, policy_hash, method)
            item["scores"][method] = {key: value for key, value in score.items() if key not in ("evaluation", "experiment")}
            print(f"{split} primary fraction {methods[method]:.2f}: S={score['S']:.6f}", flush=True)
    for split, source in sources.items():
        for name, path in source.items():
            if sha256(path) != policy["sources"][split][name]["sha256"]:
                raise ValueError("An expert prediction changed during weight selection")
    if args.existing_075_summary and args.existing_075_summary.exists():
        summary["existing_075_summary"] = {"path": str(args.existing_075_summary.resolve()),
                                           "sha256": sha256(args.existing_075_summary)}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2)+"\n")


if __name__ == "__main__":
    main()
