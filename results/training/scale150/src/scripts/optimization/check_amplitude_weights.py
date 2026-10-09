#!/usr/bin/env python3
"""Self-check of amplitude-aware frame weights against stored training flux.

For a random sample of training frames the rank-k reconstructed peak-to-peak
flux amplitude s_f is compared with the amplitude of the true training psi in
the per-shot npz files, and the weight distribution is tabulated for several
(power, cap) settings. Only training-split targets are read.

Usage:
    .venv/bin/python scripts/optimization/check_amplitude_weights.py \
        --pod-dir data/pod_history128_robust_v4 --data-dir data/advanced_history_v2 \
        [--frames 2000 --verify-frames 50 --output results/...json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from fusion.amplitude_weights import QUANTILES, amplitude_weights, frame_amplitudes
from fusion.pod import split_entries


def true_amplitudes(data_dir, entries, frame_indices):
    """Peak-to-peak of the stored training psi for global training frame indices."""
    starts = np.cumsum([0] + [entry["n_frames"] for entry in entries])
    if starts[-1] <= max(frame_indices):
        raise ValueError("Frame index exceeds the training split")
    values = np.empty(len(frame_indices), dtype=np.float64)
    for i, index in enumerate(frame_indices):
        shot = int(np.searchsorted(starts, index, side="right") - 1)
        entry = entries[shot]
        with np.load(Path(data_dir) / entry["path"], allow_pickle=False) as saved:
            psi = np.asarray(saved["psi"][index - starts[shot]], dtype=np.float64)
        values[i] = psi.max() - psi.min()
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pod-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=2000, help="Random training frames for the weight tabulation")
    parser.add_argument("--verify-frames", type=int, default=50, help="Frames compared with the stored true psi")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    entries = split_entries(args.data_dir, "train")
    with np.load(args.pod_dir / "normalization.npz") as saved:
        norm = dict(saved)
    coefficients = np.load(args.pod_dir / "train_coefficients.npy", mmap_mode="r")
    total = sum(entry["n_frames"] for entry in entries)
    if len(coefficients) != total:
        raise ValueError("POD training coefficients disagree with the training manifest")
    rng = np.random.default_rng(args.seed)
    sample = np.sort(rng.choice(total, min(args.frames, total), replace=False))
    amplitudes = frame_amplitudes(coefficients[sample], norm, torch.device("cpu"))
    verify = rng.choice(len(sample), min(args.verify_frames, len(sample)), replace=False)
    truth = true_amplitudes(args.data_dir, entries, sample[verify])
    relative = np.abs(amplitudes[verify] - truth) / truth
    report = {
        "pod_dir": str(args.pod_dir), "data_dir": str(args.data_dir), "training_frames": int(total),
        "sampled_frames": int(len(sample)), "verified_frames": int(len(verify)),
        "amplitude_quantiles": {f"q{int(round(100 * q)):02d}": float(np.quantile(amplitudes, q)) for q in QUANTILES},
        "true_amplitude_range": [float(truth.min()), float(truth.max())],
        "max_relative_discrepancy": float(relative.max()), "mean_relative_discrepancy": float(relative.mean()),
        "settings": [], "audit_labels_read": False, "validation_labels_read": False, "passed": True,
    }
    if not relative.max() < 1e-3:
        raise ValueError(f"Reconstructed amplitude differs from stored psi by {relative.max():.3e}")
    for power in (0.5, 1.0):
        for cap in (4.0, 8.0):
            weights, meta = amplitude_weights(amplitudes, power, cap)
            mean = float(weights.mean())
            if abs(mean - 1) > 1e-5:
                raise ValueError(f"Weights do not average to one: {mean}")
            report["settings"].append({
                "power": power, "max_weight": cap, "mean_weight": mean,
                "weight_quantiles": meta["normalized_weight_quantiles_used_frames"],
                "weight_max": meta["normalized_weight_max"],
                "fraction_clipped_at_max": meta["fraction_clipped_at_max_used_frames"],
                "fraction_at_unit_floor": meta["fraction_at_unit_floor_used_frames"],
                "effective_sample_fraction": meta["effective_sample_fraction_used_frames"],
                "reference_amplitude_median": meta["reference_amplitude_median"],
            })
    # Exercise the subset path: reference and rescaling over half the frames.
    subset = np.sort(rng.choice(len(sample), len(sample) // 2, replace=False))
    weights, _ = amplitude_weights(amplitudes, 1.0, 8.0, subset)
    if abs(float(weights[subset].mean()) - 1) > 1e-5 or len(weights) != len(sample):
        raise ValueError("Subset-normalized weights must average to one over the subset")
    report["subset_normalization_checked"] = True
    print(json.dumps(report, indent=2), flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
