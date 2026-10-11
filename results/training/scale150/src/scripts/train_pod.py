#!/usr/bin/env python3
"""Train POD DeepONet variants and select checkpoints using the official metric.

The POD basis and all normalization are fitted on training shots. Compact
coefficient targets allow the branch model to train on hundreds of thousands of
frames without allocating a full output grid in each optimization step.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import zipfile

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fusion.advanced_models import build_advanced_model
from fusion.pod import load_arrays, manifest_hash, read_manifest, split_entries, standardize_x, validate_splits
from scripts.train import prediction_entries, write_training_manifest


def torch_normalization(norm, device):
    return {key: torch.as_tensor(value, device=device) for key, value in norm.items()}


def reconstruct(model, outputs, norm):
    coarse = ((outputs["coefficients"] * norm["coefficient_std"]) @ norm["basis"]).reshape(-1, 65, 65) + norm["psi_mean"]
    if "flux_residual" in outputs:
        coarse = coarse + outputs["flux_residual"] * norm["psi_std"]
    if hasattr(model, "refine"):
        coarse = coarse + model.refine(coarse / norm["psi_std"], outputs["conditioning"]) * norm["psi_std"]
    return coarse


def target_arrays(pod_dir, split, device):
    return {key: torch.as_tensor(np.array(np.load(pod_dir / f"{split}_{key}.npy", mmap_mode="r")), device=device)
            for key in ("X", "coefficients", "scalars")}


def envelope_weights(device, dtype):
    with np.load(ROOT / "starter/fusion_scoring/masks/d3d_envelope.npz") as mask:
        weights = (0.1 + 0.9 * mask["mask_coarse"].astype(np.float64)) / mask["grid_R"][None, :]
    return torch.as_tensor(weights, device=device, dtype=dtype)


def band_multipliers(n_modes, spec, device=None, dtype=None):
    """Per-mode multipliers from a 'lo:hi:factor,...' spec; unlisted modes keep 1.

    The coefficient weight (std/std[0])**power is MONOTONIC in mode index, so the power family
    cannot raise modes 16-48 without also raising 48-128 -- which are close to unpredictable and
    worth 4% of the score. Sweeping the power therefore tested only a line through weight space
    (power 0 flattens everything and drops S to 0.987). This gives the middle bands their own
    factor, which is the direction the ceiling measurement points at: modes 32-48 receive 0.31%
    of the loss while holding 22% of the remaining score error.
    """
    import torch as _torch
    w = _torch.ones(n_modes, dtype=dtype or _torch.float64, device=device)
    if not spec:
        return w
    for piece in str(spec).split(","):
        piece = piece.strip()
        if not piece:
            continue
        lo, hi, factor = piece.split(":")
        lo, hi, factor = int(lo), int(hi), float(factor)
        if not 0 <= lo < hi <= n_modes:
            raise ValueError(f"band {lo}:{hi} outside 0..{n_modes}")
        if factor <= 0:
            raise ValueError(f"band factor must be positive, got {factor}")
        w[lo:hi] *= factor
    return w


def loss_matrix(norm, coefficient_power, gradient_weight, gradient_region="all", laplacian_weight=0.0,
                band_weights="", reference_rank=0):
    """Positive semidefinite loss on POD coefficient errors.

    Physical H1 term is normalized by the retained training-basis derivative
    energy. Its compression-residual cross term is not included; full-field
    Fourier fine-tuning uses native targets and exact spatial differences.
    """
    std = norm["coefficient_std"].double()
    weights = (std / std[0]).pow(coefficient_power)
    weights = weights * band_multipliers(len(weights), band_weights, weights.device, weights.dtype)
    reference_rank = reference_rank or len(weights)
    if not 0 < reference_rank <= len(weights):
        raise ValueError("Invalid loss reference rank")
    Q = torch.diag(weights / weights[:reference_rank].sum())
    basis = norm["basis"].double().reshape(-1, 65, 65) * std[:, None, None]
    dr, dz = 1.7 / 64, 3.2 / 64
    if gradient_region == "envelope":
        gz, gr = torch.gradient(basis, spacing=(dz, dr), dim=(-2, -1), edge_order=1)
        root_weights = envelope_weights(basis.device, basis.dtype).sqrt()
        gradients = torch.cat(((gr * root_weights).flatten(1), (gz * root_weights).flatten(1)), dim=1)
    else:
        gradients = torch.cat(((basis[:, :, 1:] - basis[:, :, :-1]).flatten(1) / dr,
                               (basis[:, 1:] - basis[:, :-1]).flatten(1) / dz), dim=1)
    G = gradients @ gradients.T
    Q += gradient_weight * G / torch.trace(G[:reference_rank, :reference_rank]).clamp_min(1e-12)
    if laplacian_weight:
        # Envelope-weighted Grad-Shafranov operator (toroidal current density) of the
        # scaled basis: Delta* psi = R d/dR (1/R dpsi/dR) + d2psi/dZ2, trace-normalised
        # like the H1 term so the weight is a fraction of the coefficient loss.
        with np.load(ROOT / "starter/fusion_scoring/masks/d3d_envelope.npz") as mask:
            R = torch.as_tensor(mask["grid_R"].astype(np.float64), dtype=basis.dtype, device=basis.device)
        gz, gr = torch.gradient(basis, spacing=(dz, dr), dim=(-2, -1), edge_order=1)
        gzz = torch.gradient(gz, spacing=(dz,), dim=(-2,), edge_order=1)[0]
        grr_over_r = torch.gradient(gr / R[None, None, :], spacing=(dr,), dim=(-1,), edge_order=1)[0]
        gs = R[None, None, :] * grr_over_r + gzz
        root_weights = envelope_weights(basis.device, basis.dtype).sqrt()
        L = (gs * root_weights).flatten(1)
        H = L @ L.T
        Q += laplacian_weight * H / torch.trace(H).clamp_min(1e-12)
    return Q.float()


def band_field_terms(field, target_field, targets, psi_std, options):
    """Per-frame (B,) pixel-weighted normalised squared field error of --band-loss-weight.

    sum(w * (psi_pred - psi_tgt)^2) / sum(w) / psi_std^2 with the float32 pixel
    weights w = 1 + band_weight * BAND + core_weight * CORE of
    fusion.geometry_loss.band_pixel_weights (target inside mask and peak cell).
    """
    from fusion.geometry_loss import band_pixel_weights
    weights, _, _ = band_pixel_weights(targets["inside"], targets["axis_indices"][:, 0], options["band_weight"],
                                       options["core_weight"], options["band_cells"], options["core_cells"])
    squared = ((field - target_field) / psi_std).square()
    return (squared * weights).flatten(1).sum(1) / weights.flatten(1).sum(1)


def target_field(coefficients, norm):
    """Rank-k reconstruction of label coefficients (no model refinement)."""
    return ((coefficients * norm["coefficient_std"]) @ norm["basis"]).reshape(-1, 65, 65) + norm["psi_mean"]


AUTOCAST_BF16 = False   # set from --bf16: bfloat16 autocast for the model forward only (losses stay fp32)


HEADS_W = 1.0   # set from --heads-weight in main()


def batch_loss(model, batch, norm, Q, scalar_weight, full_psi=None, field_gradient_weight=0.1, return_parts=False,
               li_module=None, li_positions=None, li_targets=None, li_weight=0.0,
               geometry_module=None, geometry_weight=0.0, exact_axis_module=None, exact_axis_weight=0.0,
               boundary_loss_weight=1.0, shape_metric_weight=0.0, axis_margin_weight=0.0, band_loss_weight=0.0, band_options=None,
               loss_parts=None, dynamic_kappa_module=None, dynamic_kappa_indices=None,
               dynamic_kappa_weight=0.0, dynamic_kappa_samples=8,
               soft_axis_module=None, soft_axis_weight=0.0, soft_axis_targets=None, soft_axis_var=None, batch_indices=None, coef_huber_rel=0.0):
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=AUTOCAST_BF16 and batch["X"].is_cuda):
        pred = model(batch["X"])
    pred = {k: v.float() for k, v in pred.items()}
    diff = pred["coefficients"] - batch["coefficients"]
    frame_weights = batch.get("frame_weights")
    def frame_mean(values):
        return values.mean() if frame_weights is None else (values * frame_weights).sum() / frame_weights.sum()
    quad = ((diff @ Q) * diff).sum(1)
    if coef_huber_rel:
        # per-frame Huber on the coefficient quadratic form: frames whose RMS error exceeds coef_huber_rel x the batch RMS
        # contribute linearly (tail-robust aggregation; the metric is still MSE, so this is a hypothesis test, not an alignment)
        delta = coef_huber_rel * quad.detach().mean().clamp_min(1e-12).sqrt(); r = quad.clamp_min(1e-12).sqrt()
        quad = torch.where(r <= delta, quad, delta * (2 * r - delta))
    loss = frame_mean(quad)
    if loss_parts is not None:
        loss_parts["coefficient"] = float(loss.detach())
    finite = torch.isfinite(batch["scalars"])
    scalar_error = pred["scalars"] - torch.nan_to_num(batch["scalars"], nan=0.0, posinf=0.0, neginf=0.0)
    scalar_weights = finite if frame_weights is None else finite * frame_weights[:, None]
    if scalar_weights.shape[1] > 2:   # extra heads: q95/betaN term exactly as with 2 scalars, extra heads add HEADS_W x their SSE
        scalar_weights = scalar_weights.float()
        scalar_count = scalar_weights[:, :2].sum()
        colw = torch.ones(scalar_weights.shape[1], device=scalar_weights.device); colw[2:] = HEADS_W
        scalar_sse = (scalar_error.square() * scalar_weights * colw).sum()
    else:
        scalar_sse = (scalar_error.square() * scalar_weights).sum()
        scalar_count = scalar_weights.sum()
    field = None
    if full_psi is not None:
        field = reconstruct(model, pred, norm)
        error = (field - full_psi) / norm["psi_std"]
        loss = loss + frame_mean(error.square().flatten(1).mean(1))
        if "field_gradient_weights" in norm:
            gz, gr = torch.gradient(error, spacing=(3.2 / 64, 1.7 / 64), dim=(-2, -1), edge_order=1)
            weights = norm["field_gradient_weights"]
            penalty = frame_mean(((gr.square() + gz.square()) * weights).flatten(1).sum(1) / weights.sum())
        else:
            gr = (error[:, :, 1:] - error[:, :, :-1]) / (1.7 / 64)
            gz = (error[:, 1:] - error[:, :-1]) / (3.2 / 64)
            penalty = frame_mean(gr.square().flatten(1).mean(1) + gz.square().flatten(1).mean(1))
        loss = loss + field_gradient_weight * penalty
    if li_module is not None and li_positions is not None and len(li_positions):
        selected_field = field[li_positions] if field is not None else reconstruct(
            model, {key: value[li_positions] for key, value in pred.items()}, norm)
        if frame_weights is None:
            loss = loss + li_weight * li_module(selected_field, li_targets)
        else:
            # Frame-weighted form of FixedContourLiLoss.forward: weighted mean of
            # the per-frame variance-normalized squared li errors.
            li_pred = li_module.predict(selected_field, li_targets)
            li_error = (li_pred - li_targets["li"].to(li_pred)).square() / li_module.li_variance.to(li_pred)
            li_weights = frame_weights[li_positions].to(li_error)
            loss = loss + li_weight * (li_error * li_weights).sum() / li_weights.sum()
        if band_loss_weight:
            # Pixel-weighted reconstructed-field loss on the li-supervised frames,
            # frame-weighted exactly like the li term (weighted mean over frames).
            band_terms = band_field_terms(selected_field, target_field(batch["coefficients"][li_positions], norm),
                                          li_targets, norm["psi_std"], band_options)
            if frame_weights is None:
                band_term = band_terms.mean()
            else:
                band_weights = frame_weights[li_positions].to(band_terms)
                band_term = (band_terms * band_weights).sum() / band_weights.sum()
            loss = loss + band_loss_weight * band_term
            if loss_parts is not None:
                loss_parts["band"] = float(band_term.detach())
        if geometry_module is not None:
            if frame_weights is None and boundary_loss_weight == 1.0 and not axis_margin_weight:
                loss = loss + geometry_weight * geometry_module(selected_field, li_targets)
            else:
                # Per-frame form of FixedGeometryLoss: frame-weighted means of the
                # axis and LCFS displacement vectors, with the LCFS part scaled by
                # --boundary-loss-weight, plus the optional peak-cell margin hinge.
                geometry = geometry_module.per_frame(selected_field, li_targets, margin=bool(axis_margin_weight))
                if frame_weights is None:
                    def geometry_mean(values):
                        return values.mean()
                else:
                    geometry_weights = frame_weights[li_positions].to(geometry["lcfs"])
                    def geometry_mean(values):
                        return (values * geometry_weights).sum() / geometry_weights.sum()
                if geometry_weight:
                    loss = loss + geometry_weight * (geometry_mean(geometry["axis"])
                                                     + boundary_loss_weight * geometry_mean(geometry["lcfs"]))
                if shape_metric_weight:
                    loss = loss + shape_metric_weight * geometry_mean(geometry["shape"])
                if axis_margin_weight:
                    loss = loss + axis_margin_weight * geometry_mean(geometry["axis_margin"])
        if exact_axis_module is not None:
            loss = loss + exact_axis_weight * exact_axis_module(selected_field, li_targets)
        if soft_axis_module is not None and soft_axis_weight and batch_indices is not None:
            # Scorer-exact differentiable axis (softmax over phi/tau) vs the SAME statistic of the true field, standardised by the
            # training variance of the true axis coordinates; frame-weighted like the li term.
            tgt = soft_axis_targets[batch_indices[li_positions]].to(selected_field)
            ok = torch.isfinite(tgt).all(1)
            if ok.any():
                ax = soft_axis_module(selected_field[ok]); err = ((ax - tgt[ok]).square() / soft_axis_var.to(ax)).sum(1)
                if frame_weights is None:
                    sa_term = err.mean()
                else:
                    sw = frame_weights[li_positions][ok].to(err); sa_term = (err * sw).sum() / sw.sum()
                loss = loss + soft_axis_weight * sa_term
                if loss_parts is not None:
                    loss_parts["soft_axis"] = float(sa_term.detach())
    if dynamic_kappa_module is not None and dynamic_kappa_weight:
        count = min(dynamic_kappa_samples, len(batch["X"]))
        selected_pred = {key: value[:count] for key, value in pred.items()}
        selected_field = reconstruct(model, selected_pred, norm)
        kappa_term = dynamic_kappa_module(
            pred["coefficients"][:count], selected_field, dynamic_kappa_indices[:count],
            frame_weights=None if frame_weights is None else frame_weights[:count],
            batch_weight_mean=None if frame_weights is None else frame_weights.mean())
        loss = loss + dynamic_kappa_weight * kappa_term
    if return_parts:
        return loss, scalar_sse, scalar_count
    return loss + scalar_weight * scalar_sse / scalar_count.clamp_min(1)


@torch.no_grad()
def band_loss_diagnostics(model, training, training_indices, norm, Q, li_store, options, batch_size, device):
    """Magnitude of the band term and of the coefficient loss on the first training batch (eval mode, no RNG)."""
    from fusion.geometry_loss import band_pixel_weights
    was_training = model.training
    model.eval()
    indices = training_indices[:batch_size]
    batch = {key: value[indices] for key, value in training.items()}
    li_positions, li_targets = li_store.batch(indices, device)
    parts = {}
    batch_loss(model, batch, norm, Q, 0.0, return_parts=True, band_loss_weight=1.0, band_options=options,
               li_module=li_store.loss_module(device), li_positions=li_positions, li_targets=li_targets, li_weight=0.0,
               loss_parts=parts)
    weights, band, core = band_pixel_weights(li_targets["inside"], li_targets["axis_indices"][:, 0], options["band_weight"],
                                             options["core_weight"], options["band_cells"], options["core_cells"])
    result = {"frames": int(len(li_positions)), "batch_frames": int(len(indices)),
              "coefficient_loss": parts.get("coefficient"), "band_loss": parts.get("band"),
              "band_to_coefficient_ratio": None if not parts.get("coefficient") else parts["band"] / parts["coefficient"],
              "band_cells_per_frame_mean": float(band.flatten(1).sum(1).float().mean()),
              "band_fraction_of_grid": float(band.float().mean()),
              "core_cells_per_frame": [int(v) for v in core.flatten(1).sum(1).unique()],
              "weights_finite": bool(torch.isfinite(weights).all()),
              "weights_min_max": [float(weights.min()), float(weights.max())],
              "mean_weight_sum_per_frame": float(weights.flatten(1).sum(1).mean())}
    if was_training:
        model.train()
    return result


@torch.no_grad()
def validation(model, arrays, norm, Q, scalar_weight, batch_size, full_psi=None, field_gradient_weight=0.1):
    model.eval()
    n = len(arrays["X"])
    total = scalar_sse = scalar_count = 0.0
    for first in range(0, n, batch_size):
        sl = slice(first, first + batch_size)
        batch = {key: value[sl] for key, value in arrays.items()}
        loss, batch_sse, batch_count = batch_loss(model, batch, norm, Q, scalar_weight, None if full_psi is None else full_psi[sl], field_gradient_weight, return_parts=True)
        total += float(loss) * len(batch["X"])
        scalar_sse += float(batch_sse)
        scalar_count += float(batch_count)
    return total / n + scalar_weight * scalar_sse / max(scalar_count, 1)


@torch.no_grad()
def write_predictions(model, norm, data_dir, split, output, batch_size, device, oracle=False):
    model.eval()
    tnorm = torch_normalization(norm, device)
    entries = prediction_entries(data_dir, split) if split == "test" else split_entries(data_dir, split)
    temporary = output.with_suffix(".tmp.npz")
    frames = 0; heads = {}
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as archive:
        for i, entry in enumerate(entries):
            with np.load(data_dir / entry["path"], allow_pickle=False) as shot:
                raw_x = shot["X"]
                if entry.get("X_extra"):
                    extra = np.load(data_dir / entry["X_extra"], allow_pickle=False)
                    if len(extra) != len(raw_x):
                        raise ValueError("History sidecar frame count differs from base features")
                    raw_x = np.concatenate((raw_x, extra), axis=1)
                x = standardize_x(raw_x, norm)
                field_parts, scalar_parts = [], []
                for first in range(0, len(x), batch_size):
                    sl = slice(first, first + batch_size)
                    if oracle:
                        # Compression diagnostic only: explicitly uses labels.
                        truth = shot["psi"][sl]
                        centered = truth.reshape(-1, 4225) - norm["psi_mean"].reshape(-1)
                        field = (centered @ norm["basis"].T @ norm["basis"]).reshape(-1, 65, 65) + norm["psi_mean"]
                        scalars = shot["scalars"][sl]
                    else:
                        pred = model(torch.as_tensor(x[sl], device=device))
                        field = reconstruct(model, pred, tnorm).cpu().numpy()
                        scalars = pred["scalars"].cpu().numpy() * norm["scalar_std"] + norm["scalar_mean"]
                    field_parts.append(field)
                    scalar_parts.append(scalars)
                field, scalars = np.concatenate(field_parts), np.concatenate(scalar_parts)
                if field.shape != (entry["n_frames"], 65, 65) or scalars.shape[0] != entry["n_frames"] or scalars.shape[1] < 2:
                    raise ValueError(f"Invalid prediction shape: {entry}")
                if not np.isfinite(field).all() or not np.isfinite(scalars).all():
                    raise ValueError(f"Nonfinite predictions: {entry}")
                for name, array in (("psirz", field), ("q95", scalars[:, 0]), ("betaN", scalars[:, 1])):
                    with archive.open(f"shot_{i:04d}_{name}.npy", "w", force_zip64=True) as stream:
                        np.lib.format.write_array(stream, array.astype(np.float32), allow_pickle=False)
                if scalars.shape[1] > 2:   # extra scalar heads go to a sidecar (<payload>_heads.npz); the payload stays standard
                    heads[f"shot_{i:04d}_heads"] = scalars[:, 2:].astype(np.float32)
                frames += len(x)
            if (i + 1) % 100 == 0:
                print(f"Predicted {split}: {i + 1}/{len(entries)} shots", flush=True)
    temporary.replace(output)
    if heads:
        np.savez(output.with_name(output.stem + "_heads.npz"), **heads)
    print(f"Wrote {output}: {len(entries)} shots, {frames} frames", flush=True)


def score_predictions(data_dir, split, pred, output, workers=4, oracle=False):
    cmd = [sys.executable, str(ROOT / "scripts/evaluate.py"), "--data-dir", str(data_dir),
           "--split", split, "--pred", str(pred), "--output", str(output), "--workers", str(workers)]
    with output.with_suffix(".log").open("w") as log:
        subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
    metrics = json.loads(output.read_text())
    if oracle:
        metrics["evaluation"]["type"] = "oracle_POD_compression_diagnostic_uses_validation_labels"
        metrics["evaluation"]["model_prediction"] = False
        output.write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"Official local {split} S={metrics['S']:.8f}, flux R²={metrics['r2_psi']:.8f}", flush=True)
    return metrics


def initialize_ridge(model, pod_dir, alpha, device):
    from scipy.linalg import solve
    X = np.load(pod_dir / "train_X.npy", mmap_mode="r")
    coefficients = np.load(pod_dir / "train_coefficients.npy", mmap_mode="r")
    scalars = np.load(pod_dir / "train_scalars.npy", mmap_mode="r")
    n_features = X.shape[1]
    gram = np.zeros((n_features + 1, n_features + 1), dtype=np.float64)
    target = np.zeros((n_features + 1, coefficients.shape[1] + 2), dtype=np.float64)
    # Scalars currently finite in the source; refuse imputation if that changes.
    if not np.isfinite(scalars).all():
        raise ValueError("Ridge warm start requires finite scalar labels; disable with --ridge-alpha -1")
    for first in range(0, len(X), 2048):
        x = np.column_stack((X[first:first + 2048], np.ones(min(2048, len(X) - first))))
        y = np.column_stack((coefficients[first:first + 2048], scalars[first:first + 2048]))
        gram += x.T @ x
        target += x.T @ y
    penalty = np.eye(n_features + 1) * alpha
    penalty[-1, -1] = 1e-10
    weights = solve(gram + penalty, target, assume_a="pos", check_finite=False)
    k = coefficients.shape[1]
    with torch.no_grad():
        for layer, chosen in ((model.linear_coefficients, slice(0, k)), (model.linear_scalars, slice(k, k + 2))):
            layer.weight.copy_(torch.as_tensor(weights[:-1, chosen].T, dtype=torch.float32, device=device))
            layer.bias.copy_(torch.as_tensor(weights[-1, chosen], dtype=torch.float32, device=device))
    print(f"Initialized linear skip with train-only ridge alpha={alpha}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=["pod_deeponet", "pod_fourier_deeponet", "pod_attention", "pod_token"],
                   default="pod_deeponet")
    p.add_argument("--attention-dim", type=int, default=64, help="pod_attention token/latent width")
    p.add_argument("--attention-latents", type=int, default=32, help="pod_attention latent query count")
    p.add_argument("--attention-heads", type=int, default=4, help="pod_attention attention heads")
    p.add_argument("--cross-layers", type=int, default=1, help="pod_attention latent-to-token blocks")
    p.add_argument("--self-layers", type=int, default=4, help="pod_attention latent self-attention blocks")
    p.add_argument("--band-weights", type=str, default="",
                   help="Per-band coefficient-loss multipliers as 'lo:hi:factor,...' (e.g. 16:32:3,32:48:6); "
                        "empty reproduces the unmodified weighting")
    p.add_argument("--warmup-epochs", type=int, default=0,
                   help="Linear LR warmup epochs before cosine annealing; 0 reproduces the untouched schedule")
    p.add_argument("--warmup-start-factor", type=float, default=0.01,
                   help="Fraction of --lr at the first warmup epoch")
    p.add_argument("--attention-tokens", type=int, default=128,
                   help="pod_attention sensor tokens; contiguous feature runs are grouped into one token each")
    p.add_argument("--data-dir", type=Path, default=Path("data/advanced"))
    p.add_argument("--pod-dir", type=Path, default=Path("data/pod"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--thomson-dropout", type=float, default=0.0,
                   help="Training-only independent probability of dropping core/edge Thomson values and availability masks")
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--weight-decay", type=float, default=0.0001)
    p.add_argument("--coefficient-power", type=float, default=1.0)
    p.add_argument("--loss-reference-rank", type=int, default=0,
                   help="Keep coefficient and H1 loss normalization fixed to this leading rank")
    p.add_argument("--amplitude-reference-pod", type=Path,
                   help="Use the same reference training-frame amplitudes in both paired rank arms")
    p.add_argument("--gradient-weight", type=float, default=0.25)
    p.add_argument("--gradient-region", choices=["auto", "all", "envelope"], default="auto",
                   help="Auto uses all-grid POD gradients and plasma-envelope Fourier gradients")
    p.add_argument("--laplacian-weight", type=float, default=0.0,
                   help="Trace-normalised weight of the envelope-weighted Grad-Shafranov-operator (current density) term of the coefficient loss")
    p.add_argument("--field-gradient-weight", type=float, default=0.1)
    p.add_argument("--scalar-weight", type=float, default=0.25)
    p.add_argument("--heads-weight", type=float, default=1.0,
                   help="per-entry loss weight of scalar heads beyond q95/betaN (views with more than 2 scalar columns); the q95/betaN term is unchanged")
    p.add_argument("--ridge-alpha", type=float, default=-1.0,
                   help="Negative disables ridge warm start; optional nonnegative penalty fits the linear skips")
    p.add_argument("--li-dir", type=Path, help="Optional fixed training-contour internal-inductance targets")
    p.add_argument("--li-weight", type=float, default=0.02)
    p.add_argument("--muon-lr-ratio", type=float, default=0.,
                   help="Positive enables the frozen six-matrix Muon partition; zero preserves native AdamW")
    p.add_argument("--dynamic-kappa-weight", type=float, default=0.0)
    p.add_argument("--dynamic-kappa-targets", type=Path)
    p.add_argument("--dynamic-kappa-every", type=int, default=4)
    p.add_argument("--dynamic-kappa-samples", type=int, default=8)
    p.add_argument("--dynamic-kappa-workers", type=int, default=2)
    p.add_argument("--dynamic-kappa-gradient-cap", type=float, default=1.0)
    p.add_argument("--li-sampled-only", action="store_true", help="Fine-tune only frames having valid training li supervision")
    p.add_argument("--soft-axis-weight", type=float, default=0.0, help="weight of the differentiable soft-argmax axis loss (li-supervised frames)")
    p.add_argument("--soft-axis-tau", type=float, default=0.01, help="softmax temperature as a fraction of psi_std")
    p.add_argument("--soft-axis-targets", type=str, default="", help=".npy (N_train, 2) soft-axis statistics of the TRUE fields (same tau), indexed by global frame")
    p.add_argument("--bf16", action="store_true", help="bfloat16 autocast for the model forward during training (transformer members)")
    p.add_argument("--coef-huber-rel", type=float, default=0.0, help="per-frame Huber on the coefficient loss: delta = value x batch RMS (0 = plain quadratic)")
    p.add_argument("--geometry-weight", type=float, default=0.0,
                   help="Optional axis/LCFS displacement surrogate weight; requires geometry-augmented --li-dir")
    p.add_argument("--boundary-loss-weight", type=float, default=1.0,
                   help="Multiplier on the LCFS displacement part of the geometry surrogate (the axis part keeps "
                        "--geometry-weight), so the two parts can be weighted independently; 1 reproduces the joint form")
    p.add_argument("--shape-metric-weight", type=float, default=0.0,
                   help="Weight of the differentiable elongation and volume surrogates, which are signed contrasts of the boundary normal displacement expressed in the metric's own units")
    p.add_argument("--boundary-extremal-weight", type=float, default=0.0,
                   help="Extremal emphasis W of the LCFS displacement loss: contour point weights 1 + W * sum of Gaussian "
                        "bumps centred on the target contour's argmax Z, argmin Z, argmax R and argmin R vertices, "
                        "normalised to mean 1; targets kappa (Z extents), triangularity and volume. 0 disables")
    p.add_argument("--boundary-extremal-sigma", type=float, default=0.06,
                   help="Arc-length width in metres of the extremal bumps (~6 of 512 points for 0.06 m)")
    p.add_argument("--axis-margin-weight", type=float, default=0.0,
                   help="Weight of the axis peak-cell margin hinge: on the 4 stored stencil neighbours of the target "
                        "peak cell require phi_pred[peak] - phi_pred[k] >= rho * (phi_gt[peak] - phi_gt[k]) with "
                        "phi = -psi, penalty mean_k relu(.)^2 / rms_target_margin^2 (reduces integer-peak-cell "
                        "switches, the main R_axis error). Requires geometry-augmented --li-dir; 0 disables")
    p.add_argument("--axis-margin-ratio", type=float, default=0.5, help="rho of the axis margin hinge, in [0, 1]")
    p.add_argument("--band-loss-weight", type=float, default=0.0,
                   help="Multiplier of the pixel-weighted reconstructed-field loss on the li-supervised frames of each "
                        "batch: loss_band = sum(w * (psi_pred - psi_tgt)^2) / sum(w) / psi_std^2 where psi_pred is the "
                        "reconstructed prediction, psi_tgt the rank-k reconstruction of the label coefficients and "
                        "w = 1 + band_weight * BAND + core_weight * CORE (float32 per pixel); BAND = "
                        "dilate(inside, band_cells) & ~erode(inside, band_cells) around the target LCFS and CORE the "
                        "(2*core_cells+1)^2 patch around the target peak cell. Frame-weighted like the li term. "
                        "Requires geometry-augmented --li-dir; 0 disables (default, bit-identical to the old path)")
    p.add_argument("--band-weight", type=float, default=8.0, help="band_weight of the BAND ring in the pixel weights")
    p.add_argument("--core-weight", type=float, default=8.0, help="core_weight of the CORE patch in the pixel weights")
    p.add_argument("--band-cells", type=int, default=2, help="Half-width in cells of the LCFS band on each side")
    p.add_argument("--core-cells", type=int, default=2, help="Half-width in cells of the axis core patch")
    p.add_argument("--frame-weighting", choices=["none", "native_count", "amplitude"], default="none",
                   help="Per-frame training loss weights: native_count uses training-only counts of original native "
                        "indices nearest each retained frame (field/scalar losses only, incompatible with --li-dir); "
                        "amplitude up-weights small-flux-amplitude target frames (ramp-down / low |Ip|) with "
                        "w_f = clip((s_ref / s_f) ** p, 1, w_max) rescaled to mean 1, where s_f is the peak-to-peak "
                        "reconstructed target flux and s_ref its median; it also weights the li loss and works with "
                        "--li-dir, --li-sampled-only and --init")
    p.add_argument("--amplitude-power", type=float, default=1.0,
                   help="Exponent p of the amplitude weights (--frame-weighting amplitude); 0 gives uniform weights")
    p.add_argument("--amplitude-max", type=float, default=8.0,
                   help="Cap w_max on the raw amplitude weight before mean-one rescaling (--frame-weighting amplitude)")
    p.add_argument("--seconds", type=int, default=3000)
    p.add_argument("--resume-training", action="store_true", help="Save and restore exact CPU state between epoch-boundary segments")
    p.add_argument("--segment-seconds", type=int, default=0, help="Pause after this allocation-local duration; requires --resume-training")
    p.add_argument("--score-every", type=int, default=25)
    p.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--init", type=Path)
    p.add_argument("--predict-only", action="store_true")
    p.add_argument("--oracle-only", action="store_true")
    p.add_argument("--split", choices=["val", "legacy_val", "audit", "test"], default="val")
    args = p.parse_args()
    global AUTOCAST_BF16
    AUTOCAST_BF16 = bool(getattr(args, 'bf16', False))
    if args.gradient_region == "auto":
        args.gradient_region = "envelope" if args.model == "pod_fourier_deeponet" else "all"
    if min(args.epochs, args.batch_size, args.seconds, args.score_every) < 1:
        p.error("epochs, batch-size, seconds and score-every must be positive")
    if not 0 <= args.thomson_dropout < 1:
        p.error("--thomson-dropout must lie in [0, 1)")
    if args.li_sampled_only and args.li_dir is None:
        p.error("--li-sampled-only requires --li-dir")
    if args.geometry_weight < 0 or (args.geometry_weight and args.li_dir is None):
        p.error("--geometry-weight must be nonnegative and requires --li-dir")
    if args.boundary_loss_weight < 0 or args.boundary_extremal_weight < 0 or args.boundary_extremal_sigma <= 0:
        p.error("--boundary-loss-weight and --boundary-extremal-weight must be nonnegative, --boundary-extremal-sigma positive")
    if (args.boundary_loss_weight != 1.0 or args.boundary_extremal_weight) and not args.geometry_weight:
        p.error("--boundary-loss-weight and --boundary-extremal-weight act on the LCFS surrogate and require --geometry-weight > 0")
    if args.axis_margin_weight < 0 or (args.axis_margin_weight and args.li_dir is None):
        p.error("--axis-margin-weight must be nonnegative and requires a geometry-augmented --li-dir")
    if not 0 <= args.axis_margin_ratio <= 1:
        p.error("--axis-margin-ratio must lie in [0, 1]")
    if args.band_loss_weight < 0 or (args.band_loss_weight and args.li_dir is None):
        p.error("--band-loss-weight must be nonnegative and requires a geometry-augmented --li-dir")
    if min(args.band_weight, args.core_weight, args.band_cells, args.core_cells) < 0:
        p.error("--band-weight, --core-weight, --band-cells and --core-cells must be nonnegative")
    if args.frame_weighting == "native_count" and args.li_dir is not None:
        p.error("Native-count ablation currently weights the field/scalar losses and requires --li-dir to be omitted")
    if args.frame_weighting == "amplitude":
        if args.amplitude_power < 0 or args.amplitude_max < 1:
            p.error("--amplitude-power must be nonnegative and --amplitude-max at least 1")
    if args.segment_seconds < 0 or args.segment_seconds and not args.resume_training:
        p.error("A nonnegative segment duration requires training continuation")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.workers)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.resume_training and device.type != "cpu":
        p.error("Exact segmented continuation is currently validated on CPU only")
    if device.type == "cuda":
        # Disable TF32 for POD basis projection, whose small residuals matter.
        torch.backends.cuda.matmul.allow_tf32 = False
    if args.predict_only:
        saved = torch.load(args.output / "best.pt", map_location="cpu", weights_only=False)
        trained_features = saved.get("run", {}).get("feature_names")
        if trained_features is not None and trained_features != read_manifest(args.data_dir)["feature_names"]:
            raise ValueError("Prediction feature order differs from the training contract")
        norm = saved["normalization"]
        model = build_advanced_model(saved["model"], **saved["model_kwargs"]).to(device)
        model.load_state_dict(saved["state_dict"])
        filename = "diii_d_public_test.npz" if args.split == "test" else f"{args.split}_predictions.npz"
        write_predictions(model, norm, args.data_dir, args.split, args.output / filename, args.batch_size, device)
        if args.split != "test":
            score_predictions(args.data_dir, args.split, args.output / filename, args.output / f"{args.split}_score.json", args.workers)
        return
    validate_splits(args.data_dir)
    basis_meta = json.loads((args.pod_dir / "basis.json").read_text())
    if basis_meta["source_manifest_sha256"] != manifest_hash(args.data_dir):
        raise ValueError("POD/prepared data manifest changed; refit using the intended selection")
    with np.load(args.pod_dir / "normalization.npz") as saved:
        norm = dict(saved)
    kwargs = dict(input_dim=len(norm["feature_indices"]), n_coefficients=len(norm["basis"]),
                  width=args.width, depth=args.depth, dropout=args.dropout)
    if len(norm["scalar_std"]) != 2:   # views with extra scalar heads (q95, betaN + consistency scalars)
        kwargs["n_scalars"] = int(len(norm["scalar_std"]))
    global HEADS_W
    HEADS_W = args.heads_weight
    if args.model == "pod_attention":
        kwargs.update(attention_dim=args.attention_dim, attention_latents=args.attention_latents,
                      attention_heads=args.attention_heads, cross_layers=args.cross_layers,
                      self_layers=args.self_layers, attention_tokens=args.attention_tokens)
    if args.model == "pod_token":
        _names = read_manifest(args.data_dir)["feature_names"]
        kwargs.update(feature_names=[_names[int(i)] for i in norm["feature_indices"]], token_dim=args.attention_dim, heads=args.attention_heads)
    model = build_advanced_model(args.model, **kwargs).to(device)
    if args.oracle_only:
        if args.split in ("test", "audit"):
            raise ValueError("Oracle diagnostic is restricted to validation, never test/audit")
        path = args.output / "oracle_compression_predictions.npz"
        write_predictions(model, norm, args.data_dir, args.split, path, args.batch_size, device, oracle=True)
        score_predictions(args.data_dir, args.split, path, args.output / "oracle_compression_score.json", args.workers, oracle=True)
        return
    started = time.monotonic()
    write_training_manifest(args.data_dir, args.output)
    if args.init:
        saved = torch.load(args.init, map_location="cpu", weights_only=False)
        for key in ("basis", "feature_indices", "x_mean", "x_std", "coefficient_std", "psi_mean", "psi_std", "scalar_mean", "scalar_std"):
            if not np.array_equal(saved["normalization"][key], norm[key]):
                raise ValueError(f"Warm-start normalization differs: {key}")
        missing, unexpected = model.load_state_dict(saved["state_dict"], strict=False)
        if unexpected or any(not key.startswith(("refiner.", "conditioning_head.")) for key in missing):
            raise ValueError(f"Incompatible warm start: missing={missing}, unexpected={unexpected}")
        print(f"Warm started from {args.init}", flush=True)
    elif args.ridge_alpha >= 0:
        initialize_ridge(model, args.pod_dir, args.ridge_alpha, device)
    training, valid = target_arrays(args.pod_dir, "train", device), target_arrays(args.pod_dir, "val", device)
    frame_weighting = None
    if args.frame_weighting == "native_count":
        from fusion.frame_weights import training_native_weights
        frame_weights, native_counts, frame_weighting = training_native_weights(args.data_dir)
        training["frame_weights"] = torch.as_tensor(frame_weights, device=device)
        np.save(args.output / "training_native_counts.npy", native_counts, allow_pickle=False)
    ts_plans = None
    if args.thomson_dropout:
        from fusion.augmentation import thomson_dropout_plan, drop_thomson_modalities
        ts_plans = thomson_dropout_plan(read_manifest(args.data_dir)["feature_names"], norm, device)
        if not ts_plans:
            raise ValueError("Thomson dropout requested without Thomson measurement features")
    tnorm = torch_normalization(norm, device)
    Q = loss_matrix(tnorm, args.coefficient_power, args.gradient_weight, args.gradient_region,
                    args.laplacian_weight, args.band_weights, args.loss_reference_rank)
    if args.gradient_region == "envelope":
        tnorm["field_gradient_weights"] = envelope_weights(device, torch.float32)
    li_store = li_module = geometry_module = None
    training_indices = torch.arange(len(training["X"]), device=device)
    if args.li_dir:
        from fusion.li_loss import LiTargetStore
        if args.geometry_weight or args.axis_margin_weight or args.band_loss_weight:
            from fusion.geometry_loss import GeometryTargetStore
            li_store = GeometryTargetStore(args.li_dir)
            if args.geometry_weight or args.axis_margin_weight:
                geometry_module = li_store.geometry_loss_module(
                    device, boundary_extremal_weight=args.boundary_extremal_weight,
                    boundary_extremal_sigma=args.boundary_extremal_sigma, axis_margin_ratio=args.axis_margin_ratio,
                    axis_margin=bool(args.axis_margin_weight))
                if args.axis_margin_weight:
                    print(json.dumps({"axis_margin_scale_rms_psi": geometry_module.margin_scale,
                                      "axis_margin_neighbours": 4}), flush=True)
        else:
            li_store = LiTargetStore(args.li_dir)
        selected_hash = hashlib.sha256(json.dumps(split_entries(args.data_dir, "train"), sort_keys=True).encode()).hexdigest()
        if selected_hash != li_store.metadata["source_training_entries_sha256"]:
            raise ValueError("Internal-inductance target indexing differs from training entries")
        li_module = li_store.loss_module(device)
    soft_axis_module = soft_axis_targets_t = soft_axis_var = None
    if args.soft_axis_weight:
        from fusion.soft_geometry import SoftAxis
        with np.load(ROOT / "starter/fusion_scoring/masks/d3d_envelope.npz") as _m:
            _R, _Z, _mask = _m["grid_R"].astype(np.float64), _m["grid_Z"].astype(np.float64), _m["mask_coarse"].astype(bool)
        soft_axis_module = SoftAxis(_R, _Z, _mask, -1.0, args.soft_axis_tau * float(norm["psi_std"])).to(device)
        _t = np.load(args.soft_axis_targets); soft_axis_targets_t = torch.as_tensor(_t, device=device)
        _ok = np.isfinite(_t).all(1); soft_axis_var = torch.as_tensor(_t[_ok].var(0), device=device)
        print(json.dumps({"soft_axis": {"tau": args.soft_axis_tau, "targets": args.soft_axis_targets, "var": _t[_ok].var(0).tolist(), "frames": int(_ok.sum())}}), flush=True)
        if args.li_sampled_only:
            training_indices = torch.as_tensor(np.flatnonzero(li_store.lookup >= 0), device=device)
        print(f"Training li supervision: {len(np.flatnonzero(li_store.lookup >= 0))} native training frames", flush=True)
    if args.frame_weighting == "amplitude":
        # Computed after the li subset is known so the reference median and the
        # mean-one rescaling cover exactly the frames optimized each epoch.
        from fusion.amplitude_weights import amplitude_weights, frame_amplitudes
        amplitude_started = time.monotonic()
        amplitude_pod = args.amplitude_reference_pod or args.pod_dir
        if args.amplitude_reference_pod:
            with np.load(amplitude_pod / "normalization.npz") as saved:
                amplitude_norm = dict(saved)
            reference_meta = json.loads((amplitude_pod / "basis.json").read_text())
            if reference_meta["source_manifest_sha256"] != manifest_hash(args.data_dir):
                raise ValueError("Amplitude reference training manifest mismatch")
        else:
            amplitude_norm = norm
        amplitudes = frame_amplitudes(np.load(amplitude_pod / "train_coefficients.npy", mmap_mode="r"), amplitude_norm, device)
        frame_weights, frame_weighting = amplitude_weights(amplitudes, args.amplitude_power, args.amplitude_max,
                                                           training_indices.cpu().numpy())
        frame_weighting["seconds"] = time.monotonic() - amplitude_started
        training["frame_weights"] = torch.as_tensor(frame_weights, device=device)
        np.save(args.output / "frame_amplitude.npy", amplitudes, allow_pickle=False)
        print(json.dumps({"frame_weighting": frame_weighting}), flush=True)
    train_psi = val_psi = None
    if args.model == "pod_fourier_deeponet":
        train_psi = torch.as_tensor(np.array(load_arrays(args.data_dir, "train", ("psi",))["psi"]), device=device)
        val_psi = torch.as_tensor(np.array(load_arrays(args.data_dir, "val", ("psi",))["psi"]), device=device)
    dynamic_kappa = None
    if args.dynamic_kappa_weight:
        if (args.model != "pod_deeponet" or not args.dynamic_kappa_targets or
                min(args.dynamic_kappa_every, args.dynamic_kappa_samples, args.dynamic_kappa_workers) <= 0):
            raise ValueError("Dynamic kappa requires pod_deeponet, targets and positive sampling settings")
        from fusion.dynamic_contour_loss import DynamicKappaLoss
        dynamic_kappa = DynamicKappaLoss(
            norm, args.dynamic_kappa_targets, ROOT / "starter/fusion_scoring/masks/d3d_envelope.npz",
            workers=args.dynamic_kappa_workers, gradient_cap=args.dynamic_kappa_gradient_cap,
            status_path=args.output / "dynamic_kappa_status.json")
        target_hash = hashlib.sha256(json.dumps(split_entries(args.data_dir, "train"), sort_keys=True).encode()).hexdigest()
        if target_hash != dynamic_kappa.metadata["source_training_entries_sha256"]:
            raise ValueError("Dynamic kappa target indexing differs from training entries")
    # Rank changes consume different amounts of RNG during model construction.
    # Reset here so matched arms see identical frame orders and Thomson dropout.
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    from fusion.muon_optimizer import make_optimizer_and_scheduler
    optimizer, scheduler = make_optimizer_and_scheduler(model, args)
    # Optimizer construction must not influence the paired permutation/dropout.
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    run = {"model": args.model, "model_kwargs": kwargs, "parameters": sum(p.numel() for p in model.parameters()),
           "train_frames": len(training["X"]), "training_frames_per_epoch": len(training_indices),
           "val_frames": len(valid["X"]), "basis": basis_meta,
           "device": str(device), "gpu": torch.cuda.get_device_name() if device.type == "cuda" else None,
           "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
           "checkpoint_selection": "maximum full official composite on validation shots at scheduled epochs",
           "started_utc": datetime.now(timezone.utc).isoformat(), "audit_labels_read": False,
           "feature_names": read_manifest(args.data_dir)["feature_names"],
           "li_supervision": li_store.metadata if li_store else None,
           "frame_weighting": frame_weighting,
           "source_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                             for path in (ROOT / "fusion/pod.py", ROOT / "fusion/advanced_models.py", ROOT / "scripts/train_pod.py",
                                          *([ROOT / "fusion/augmentation.py"] if args.thomson_dropout else []),
                                          *([ROOT / "fusion/frame_weights.py"] if args.frame_weighting == "native_count" else []),
                                          *([ROOT / "fusion/amplitude_weights.py"] if args.frame_weighting == "amplitude" else []),
                                          *([ROOT / "fusion/geometry_loss.py", ROOT / "fusion/li_loss.py"] if args.geometry_weight or args.axis_margin_weight or args.band_loss_weight else []))}}
    if args.muon_lr_ratio:
        from fusion.muon_optimizer import MUON_NAMES
        run["muon"] = {"hidden_parameters": list(MUON_NAMES), "lr_ratio": args.muon_lr_ratio,
                       "muon_lr_initial": args.lr * args.muon_lr_ratio,
                       "muon_weight_decay": args.weight_decay / args.muon_lr_ratio,
                       "adamw_betas": [.9, .999], "adamw_eps": 1e-8,
                       "momentum": .95, "ns_steps": 5, "nesterov": True,
                       "schedule": "unchanged native AdamW cosine; Muon LR tracks its fixed ratio"}
        for name in ("fusion/muon_optimizer.py", "fusion/vendor/muon_upstream.py"):
            run["source_sha256"][name] = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
    if dynamic_kappa is not None:
        run["dynamic_kappa"] = {
            "weight": args.dynamic_kappa_weight, "every": args.dynamic_kappa_every,
            "samples": args.dynamic_kappa_samples, "workers": args.dynamic_kappa_workers,
            "coefficient_gradient_norm_cap": args.dynamic_kappa_gradient_cap,
            "selection": "First eight frames of every fourth shuffled batch; no extra RNG",
            "weight_correction": "Multiply auxiliary weight by cadence; selected amplitude weights normalized by full-batch mean",
            "targets": dynamic_kappa.metadata}
        for name in ("contour_replay.py", "dynamic_contour_loss.py"):
            run["source_sha256"]["fusion/" + name] = hashlib.sha256((ROOT / "fusion" / name).read_bytes()).hexdigest()
    band_options = None
    if args.band_loss_weight:
        band_options = {"band_weight": args.band_weight, "core_weight": args.core_weight,
                        "band_cells": args.band_cells, "core_cells": args.core_cells}
        run["band_loss"] = dict(band_options, band_loss_weight=args.band_loss_weight, frame_weighted=args.frame_weighting != "none",
                                initial=band_loss_diagnostics(model, training, training_indices, tnorm, Q, li_store, band_options,
                                                              args.batch_size, device))
        print(json.dumps({"band_loss": run["band_loss"]}), flush=True)
    if geometry_module is not None:
        run["geometry_options"] = {"boundary_loss_weight": args.boundary_loss_weight,
                                   "boundary_extremal_weight": args.boundary_extremal_weight,
                                   "boundary_extremal_sigma": args.boundary_extremal_sigma,
                                   "axis_margin_weight": args.axis_margin_weight, "axis_margin_ratio": args.axis_margin_ratio,
                                   "axis_margin_scale_rms_psi": geometry_module.margin_scale, "axis_margin_neighbours": 4,
                                   "frame_weighted": args.frame_weighting != "none"}
    if args.resume_training:
        run["source_sha256"]["fusion/training_resume.py"] = hashlib.sha256((ROOT / "fusion/training_resume.py").read_bytes()).hexdigest()
        run["initial_checkpoint_sha256"] = hashlib.sha256(args.init.read_bytes()).hexdigest() if args.init else None
    history, best_score, best_loss = [], -float("inf"), float("inf")
    first_epoch, elapsed_before = 0, 0.
    resume_path = args.output / "training_state.pt"
    if args.resume_training and resume_path.exists():
        from fusion.training_resume import restore_training_state
        state = restore_training_state(resume_path, model, optimizer, scheduler, norm, run)
        history, best_score, best_loss = state["history"], state["best_score"], state["best_loss"]
        first_epoch, elapsed_before = history[-1]["epoch"]+1, state["run"]["elapsed_seconds"]
        run["started_utc"] = state["run"]["started_utc"]
        if first_epoch > args.epochs or state["run"].get("status") == "completed":
            raise ValueError("This training run is already complete")
        print(json.dumps({"resuming_at_epoch": first_epoch, "elapsed_training_seconds": elapsed_before}), flush=True)
    run["status"] = "running"
    (args.output / "run.json").write_text(json.dumps(run, indent=2) + "\n")
    print(json.dumps({key: value for key, value in run.items() if key != "feature_names"}), flush=True)
    for epoch in range(first_epoch, args.epochs + 1):
        train_loss = 0.0
        train_parts = {}
        if epoch:
            model.train()
            order = training_indices[torch.randperm(len(training_indices), device=device)]
            for batch_index, indices in enumerate(order.split(args.batch_size)):
                batch = {key: value[indices] for key, value in training.items()}
                if ts_plans:
                    batch["X"] = drop_thomson_modalities(batch["X"], ts_plans, args.thomson_dropout)
                optimizer.zero_grad(set_to_none=True)
                li_positions, li_targets = li_store.batch(indices, device) if li_store else (None, None)
                parts = {} if band_options else None
                loss = batch_loss(model, batch, tnorm, Q, args.scalar_weight,
                                  None if train_psi is None else train_psi[indices], args.field_gradient_weight,
                                  li_module=li_module, li_positions=li_positions, li_targets=li_targets, li_weight=args.li_weight,
                                  geometry_module=geometry_module, geometry_weight=args.geometry_weight,
                                  boundary_loss_weight=args.boundary_loss_weight, shape_metric_weight=args.shape_metric_weight, axis_margin_weight=args.axis_margin_weight,
                                  band_loss_weight=args.band_loss_weight, band_options=band_options, loss_parts=parts,
                                  dynamic_kappa_module=dynamic_kappa if batch_index % args.dynamic_kappa_every == 0 else None,
                                  dynamic_kappa_indices=indices,
                                  dynamic_kappa_weight=args.dynamic_kappa_weight * args.dynamic_kappa_every,
                                  dynamic_kappa_samples=args.dynamic_kappa_samples,
                                  soft_axis_module=soft_axis_module, soft_axis_weight=args.soft_axis_weight,
                                  soft_axis_targets=soft_axis_targets_t, soft_axis_var=soft_axis_var, batch_indices=indices,
                                  coef_huber_rel=args.coef_huber_rel)
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite training loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=bool(args.muon_lr_ratio))
                optimizer.step()
                train_loss += float(loss.detach()) * len(indices)
                if parts:
                    for key, value in parts.items():
                        train_parts[key] = train_parts.get(key, 0.0) + value * len(indices)
            scheduler.step()
        val_loss = validation(model, valid, tnorm, Q, args.scalar_weight, args.batch_size, val_psi, args.field_gradient_weight)
        record = {"epoch": epoch, "train_loss": train_loss / len(training_indices), "val_loss": val_loss,
                  "elapsed_seconds": elapsed_before + time.monotonic() - started, "lr": scheduler.get_last_lr()[0]}
        if dynamic_kappa is not None:
            dynamic_kappa.write_status()
            record["dynamic_kappa"] = dynamic_kappa.status()
        if band_options and epoch:
            # Unweighted magnitudes of the coefficient quadratic form and of the band term
            # (frame-weighted batch means, averaged over the epoch), for --band-loss-weight tuning.
            record.update({f"train_{key}_loss": value / len(training_indices) for key, value in train_parts.items()})
        checkpoint = dict(model=args.model, model_kwargs=kwargs, normalization=norm,
                          state_dict=model.state_dict(), epoch=epoch, run=run)
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(checkpoint, args.output / "best_loss.pt")
        final_epoch = epoch == args.epochs or record["elapsed_seconds"] >= args.seconds
        if epoch % args.score_every == 0 or final_epoch:
            pred_path = args.output / "candidate_val_predictions.npz"
            score_path = args.output / "candidate_val_score.json"
            write_predictions(model, norm, args.data_dir, "val", pred_path, args.batch_size, device)
            metrics = score_predictions(args.data_dir, "val", pred_path, score_path, args.workers)
            record["S"] = metrics["S"]
            checkpoint["validation_score"] = metrics["S"]
            snapshot_dir = args.output / "checkpoints"
            snapshot_dir.mkdir(exist_ok=True)
            torch.save(checkpoint, snapshot_dir / f"epoch_{epoch:04d}.pt")
            if metrics["S"] > best_score:
                best_score = metrics["S"]
                torch.save(checkpoint, args.output / "best.pt")
                pred_path.replace(args.output / "val_predictions.npz")
                score_path.replace(args.output / "val_score.json")
        history.append(record)
        (args.output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        if epoch % 5 == 0 or final_epoch:
            print(json.dumps(record), flush=True)
        if final_epoch:
            break
        if args.segment_seconds and time.monotonic()-started >= args.segment_seconds:
            from fusion.training_resume import save_training_state
            run.update(status="paused", completed_epochs=epoch,
                       elapsed_seconds=elapsed_before + time.monotonic()-started, best_validation_score=best_score)
            save_training_state(resume_path, model, optimizer, scheduler, norm, run, history, best_score, best_loss)
            (args.output / "run.json").write_text(json.dumps(run, indent=2) + "\n")
            print(json.dumps({"paused_after_epoch": epoch, "scheduled_epochs": args.epochs, "exact_state_saved": True}), flush=True)
            return
    if dynamic_kappa is not None:
        dynamic_kappa.close()
    run.update(completed_epochs=epoch, elapsed_seconds=elapsed_before + time.monotonic() - started, best_validation_score=best_score,
               status="completed")
    if args.resume_training:
        from fusion.training_resume import save_training_state
        save_training_state(resume_path, model, optimizer, scheduler, norm, run, history, best_score, best_loss)
    (args.output / "run.json").write_text(json.dumps(run, indent=2) + "\n")
    saved = torch.load(args.output / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(saved["state_dict"])
    try:
        legacy = split_entries(args.data_dir, "legacy_val")
    except KeyError:
        legacy = []
    if legacy:
        pred_path = args.output / "legacy_val_predictions.npz"
        write_predictions(model, norm, args.data_dir, "legacy_val", pred_path, args.batch_size, device)
        score_predictions(args.data_dir, "legacy_val", pred_path, args.output / "legacy_val_score.json", args.workers)
    print(f"Completed; best official local validation S={best_score:.8f}", flush=True)


if __name__ == "__main__":
    main()
