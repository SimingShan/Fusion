#!/usr/bin/env python3
"""Stream the validated scaled-li/robust-history ensemble without hidden labels."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time
import zipfile

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from fusion.advanced_models import build_advanced_model
from fusion.pod import read_manifest, standardize_x
from scripts.train import prediction_entries
from scripts.train_pod import reconstruct, torch_normalization
from scripts.optimization.quality_gate import feature_columns, gate_from_inputs, write_array, sha256

PRIMARY_WEIGHT = 0.75


def load_expert(checkpoint, data_dir, device):
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    names = read_manifest(data_dir)["feature_names"]
    if saved.get("run", {}).get("feature_names") != names:
        raise ValueError(f"Feature contract differs from training for {checkpoint}")
    model = build_advanced_model(saved["model"], **saved["model_kwargs"]).to(device)
    model.load_state_dict(saved["state_dict"])
    model.eval()
    norm = saved["normalization"]
    return model, norm, torch_normalization(norm, device), names, saved["epoch"]


def input_arrays(data_dir, entry):
    with np.load(data_dir / entry["path"], allow_pickle=False) as shot:
        # Targets, if present in a validation cache, are never accessed here.
        X, times, frames = shot["X"], shot["times"], shot["frame_indices"]
    if entry.get("X_extra"):
        extra = np.load(data_dir / entry["X_extra"], allow_pickle=False)
        if len(extra) != len(X):
            raise ValueError("History sidecar frame count differs from the base shot")
        X = np.concatenate((X, extra), axis=1)
    if len(X) != entry["n_frames"]:
        raise ValueError("Input frame count differs from manifest")
    return X, times, frames


@torch.no_grad()
def predict(expert, raw_X, batch_size, device):
    model, norm, tnorm, _, _ = expert
    X = standardize_x(raw_X, norm)
    flux, scalars = [], []
    for first in range(0, len(X), batch_size):
        output = model(torch.as_tensor(X[first:first + batch_size], device=device))
        flux.append(reconstruct(model, output, tnorm).cpu().numpy())
        scalars.append(output["scalars"].cpu().numpy() * norm["scalar_std"] + norm["scalar_mean"])
    return np.concatenate(flux), np.concatenate(scalars)


def export(args):
    started = time.monotonic()
    torch.set_num_threads(args.workers)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device(args.device)
    primary_weight = args.primary_weight
    args.output.mkdir(parents=True, exist_ok=True)
    experts = [load_expert(path, data_dir, device) for path, data_dir in
               ((args.primary, args.primary_data), (args.secondary, args.secondary_data))]
    manifests = [read_manifest(directory) for directory in (args.primary_data, args.secondary_data)]
    entries = [prediction_entries(directory, "test") if args.split == "test" else manifest[args.split]
               for directory, manifest in zip((args.primary_data, args.secondary_data), manifests)]
    if len(entries[0]) != len(entries[1]) or not entries[0]:
        raise ValueError("Expert caches contain different shot selections")
    for first, second in zip(*entries):
        for key in ("shot_index", "shot_id", "shard", "row", "n_frames", "raw_frames"):
            if first[key] != second[key]:
                raise ValueError(f"Expert cache shot identity disagrees: {key}")
    columns = feature_columns(experts[0][3])
    source_hashes = {"primary": sha256(args.primary), "secondary": sha256(args.secondary)}
    filename = "diii_d_public_test.npz" if args.split == "test" else f"{args.split}_predictions.npz"
    output = args.output / filename
    temporary = output.with_suffix(".tmp.npz")
    report = {"status": "running", "created_utc": datetime.now(timezone.utc).isoformat(),
              "split": args.split, "primary": str(args.primary.resolve()), "secondary": str(args.secondary.resolve()),
              "source_checkpoint_sha256": source_hashes, "primary_epoch": experts[0][4], "secondary_epoch": experts[1][4],
              "primary_data": str(args.primary_data.resolve()), "secondary_data": str(args.secondary_data.resolve()),
              "source_manifest_sha256": [sha256(directory / "manifest.json") for directory in (args.primary_data, args.secondary_data)],
              "primary_weight": primary_weight, "gate_threshold": 0.1,
              "gate_rule": f"secondary alone when at most 10% of real edge channels have positive observed Te and ne; otherwise {primary_weight} primary + {1-primary_weight} secondary",
              "threshold_or_weight_search": args.validation_tuned_weight, "targets_read": False, "audit_labels_read": False,
              "official_submission": False, "source_code_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in
                   (Path(__file__).resolve(), ROOT / "scripts/train_pod.py", ROOT / "fusion/pod.py", ROOT / "fusion/advanced_models.py", ROOT / "fusion/input_derived.py", ROOT / "scripts/optimization/quality_gate.py")},
              "shots": len(entries[0]), "frames": 0, "gated_frames": 0, "gated_shots": 0}
    (args.output / "export.json").write_text(json.dumps(report, indent=2) + "\n")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as archive:
        for i, (first, second) in enumerate(zip(*entries)):
            X1, times1, frames1 = input_arrays(args.primary_data, first)
            X2, times2, frames2 = input_arrays(args.secondary_data, second)
            if not np.array_equal(times1, times2) or not np.array_equal(frames1, frames2):
                raise ValueError("Expert caches have different native query timestamps")
            gated, _, _, _ = gate_from_inputs(X1, columns)
            psi1, scalar1 = predict(experts[0], X1, args.batch_size, device)
            psi2, scalar2 = predict(experts[1], X2, args.batch_size, device)
            psi = np.where(gated[:, None, None], psi2, primary_weight * psi1 + (1 - primary_weight) * psi2)
            scalars = np.where(gated[:, None], scalar2, primary_weight * scalar1 + (1 - primary_weight) * scalar2)
            if psi.shape != (len(times1), 65, 65) or scalars.shape != (len(times1), 2):
                raise ValueError("Ensemble prediction shape differs from the native output contract")
            for field, values in (("psirz", psi), ("q95", scalars[:, 0]), ("betaN", scalars[:, 1])):
                write_array(archive, f"shot_{i:04d}_{field}", values)
            report["frames"] += len(times1)
            report["gated_frames"] += int(gated.sum())
            report["gated_shots"] += int(gated.any())
            if (i + 1) % 100 == 0 or i + 1 == len(entries[0]):
                print(f"Ensemble export: {i + 1}/{len(entries[0])} shots, {report['frames']} frames", flush=True)
    if source_hashes != {"primary": sha256(args.primary), "secondary": sha256(args.secondary)}:
        raise ValueError("An expert checkpoint changed during export")
    os.replace(temporary, output)
    report.update(status="complete", prediction=str(output.resolve()), prediction_sha256=sha256(output),
                  prediction_bytes=output.stat().st_size, elapsed_seconds=time.monotonic() - started)
    (args.output / "export.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("status", "shots", "frames", "gated_frames", "elapsed_seconds")}), flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", type=Path, default=Path("results/optimization/pod_scaled_li_v2/best.pt"))
    parser.add_argument("--secondary", type=Path, default=Path("results/optimization/pod_history64_robust_v3/best.pt"))
    parser.add_argument("--primary-data", type=Path, default=Path("data/advanced_test_v1"))
    parser.add_argument("--secondary-data", type=Path, default=Path("data/advanced_history_test_v2"))
    parser.add_argument("--output", type=Path, default=Path("results/optimization/submission_scaled_robust_v5"))
    parser.add_argument("--split", choices=("val", "legacy_val", "test"), default="test")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--primary-weight", type=float, default=PRIMARY_WEIGHT)
    parser.add_argument("--validation-tuned-weight", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1 or args.workers < 1:
        parser.error("Batch size and workers must be positive")
    if not 0 <= args.primary_weight <= 1:
        parser.error("Primary weight must be between zero and one")
    export(args)


if __name__ == "__main__":
    main()
