#!/usr/bin/env python3
"""Per-term training loss magnitudes of a checkpoint on li/geometry-supervised training batches.

Reports unweighted means (and amplitude-weighted means when requested) of the
coefficient loss, scalar loss, li term, geometry axis / LCFS displacement terms
(plain and extremal-emphasised) and the axis peak-cell margin hinge, so that the
loss coefficients can be chosen at a sane relative scale. Training data only.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("train_pod", ROOT / "scripts/train_pod.py")
train_pod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_pod)
from fusion.advanced_models import build_advanced_model
from fusion.geometry_loss import GeometryTargetStore


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--pod-dir", type=Path, required=True)
    p.add_argument("--li-dir", type=Path, required=True)
    p.add_argument("--batches", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--boundary-extremal-weight", type=float, default=4.0)
    p.add_argument("--boundary-extremal-sigma", type=float, default=0.06)
    p.add_argument("--axis-margin-ratio", type=float, default=0.5)
    p.add_argument("--gradient-region", default="envelope")
    p.add_argument("--amplitude-power", type=float, default=1.0)
    p.add_argument("--amplitude-max", type=float, default=8.0)
    p.add_argument("--seed", type=int, default=2029)
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    torch.set_num_threads(args.workers)
    device = torch.device("cpu")
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = build_advanced_model(saved["model"], **saved["model_kwargs"]).to(device)
    model.load_state_dict(saved["state_dict"]); model.eval()
    with np.load(args.pod_dir / "normalization.npz") as f:
        norm = dict(f)
    tnorm = train_pod.torch_normalization(norm, device)
    Q = train_pod.loss_matrix(tnorm, 1.0, 0.25, args.gradient_region)
    if args.gradient_region == "envelope":
        tnorm["field_gradient_weights"] = train_pod.envelope_weights(device, torch.float32)
    store = GeometryTargetStore(args.li_dir)
    plain = store.geometry_loss_module(device, axis_margin=True, axis_margin_ratio=args.axis_margin_ratio)
    emphasised = store.geometry_loss_module(device, boundary_extremal_weight=args.boundary_extremal_weight,
                                            boundary_extremal_sigma=args.boundary_extremal_sigma)
    li_module = store.loss_module(device)
    coefficients = np.load(args.pod_dir / "train_coefficients.npy", mmap_mode="r")
    X = np.load(args.pod_dir / "train_X.npy", mmap_mode="r")
    scalars = np.load(args.pod_dir / "train_scalars.npy", mmap_mode="r")
    supervised = np.flatnonzero(store.lookup >= 0)
    from fusion.amplitude_weights import amplitude_weights, frame_amplitudes
    amplitudes = frame_amplitudes(coefficients, norm, device)
    weights, _ = amplitude_weights(amplitudes, args.amplitude_power, args.amplitude_max, supervised)
    rng = np.random.default_rng(args.seed)
    sums = {}
    weighted = {}
    total_weight = 0.0
    with torch.no_grad():
        for _ in range(args.batches):
            idx = np.sort(rng.choice(supervised, args.batch_size, replace=False))
            batch = {"X": torch.as_tensor(np.asarray(X[idx])), "coefficients": torch.as_tensor(np.asarray(coefficients[idx])),
                     "scalars": torch.as_tensor(np.asarray(scalars[idx]))}
            pred = model(batch["X"])
            diff = pred["coefficients"] - batch["coefficients"]
            terms = {"coefficient": ((diff @ Q) * diff).sum(1)}
            finite = torch.isfinite(batch["scalars"])
            err = pred["scalars"] - torch.nan_to_num(batch["scalars"])
            terms["scalar_x0.25"] = 0.25 * (err.square() * finite).sum(1) / finite.sum(1).clamp_min(1)
            positions, targets = store.batch(torch.as_tensor(idx), device)
            field = train_pod.reconstruct(model, {k: v[positions] for k, v in pred.items()}, tnorm)
            li_pred = li_module.predict(field, targets)
            terms["li_norm"] = (li_pred - targets["li"].to(li_pred)).square() / li_module.li_variance.to(li_pred)
            g = plain.per_frame(field, targets, margin=True)
            terms["geometry_axis"] = g["axis"]; terms["geometry_lcfs"] = g["lcfs"]
            terms["axis_margin_hinge"] = g["axis_margin"]
            terms["axis_margin_violation_frac"] = (g["axis_margin"] > 0).double()
            terms["geometry_lcfs_extremal"] = emphasised.per_frame(field, targets)["lcfs"]
            w = torch.as_tensor(weights[idx], dtype=torch.float64)
            for k, v in terms.items():
                v = v.double()
                sums[k] = sums.get(k, 0.0) + float(v.sum())
                weighted[k] = weighted.get(k, 0.0) + float((v * w).sum())
            total_weight += float(w.sum())
    n = args.batches * args.batch_size
    report = {"checkpoint": str(args.checkpoint), "frames": n, "axis_margin_scale_rms_psi": plain.margin_scale,
              "unweighted_mean": {k: v / n for k, v in sums.items()},
              "amplitude_weighted_mean": {k: v / total_weight for k, v in weighted.items()},
              "settings": {k: getattr(args, k) for k in ("boundary_extremal_weight", "boundary_extremal_sigma", "axis_margin_ratio",
                                                          "amplitude_power", "amplitude_max", "batches", "batch_size")}}
    text = json.dumps(report, indent=2)
    print(text, flush=True)
    if args.output:
        args.output.write_text(text + "\n")


if __name__ == "__main__":
    main()
