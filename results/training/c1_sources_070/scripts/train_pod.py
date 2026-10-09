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


def loss_matrix(norm, coefficient_power, gradient_weight, gradient_region="all"):
    """Positive semidefinite loss on POD coefficient errors.

    Physical H1 term is normalized by the retained training-basis derivative
    energy. Its compression-residual cross term is not included; full-field
    Fourier fine-tuning uses native targets and exact spatial differences.
    """
    std = norm["coefficient_std"].double()
    weights = (std / std[0]).pow(coefficient_power)
    Q = torch.diag(weights / weights.sum())
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
    Q += gradient_weight * G / torch.trace(G).clamp_min(1e-12)
    return Q.float()


def batch_loss(model, batch, norm, Q, scalar_weight, full_psi=None, field_gradient_weight=0.1, return_parts=False,
               li_module=None, li_positions=None, li_targets=None, li_weight=0.0,
               geometry_module=None, geometry_weight=0.0, exact_axis_module=None, exact_axis_weight=0.0):
    pred = model(batch["X"])
    diff = pred["coefficients"] - batch["coefficients"]
    frame_weights = batch.get("frame_weights")
    def frame_mean(values):
        return values.mean() if frame_weights is None else (values * frame_weights).sum() / frame_weights.sum()
    loss = frame_mean(((diff @ Q) * diff).sum(1))
    finite = torch.isfinite(batch["scalars"])
    scalar_error = pred["scalars"] - torch.nan_to_num(batch["scalars"], nan=0.0, posinf=0.0, neginf=0.0)
    scalar_weights = finite if frame_weights is None else finite * frame_weights[:, None]
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
        loss = loss + li_weight * li_module(selected_field, li_targets)
        if geometry_module is not None:
            loss = loss + geometry_weight * geometry_module(selected_field, li_targets)
        if exact_axis_module is not None:
            loss = loss + exact_axis_weight * exact_axis_module(selected_field, li_targets)
    if return_parts:
        return loss, scalar_sse, scalar_count
    return loss + scalar_weight * scalar_sse / scalar_count.clamp_min(1)


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
    frames = 0
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
                if field.shape != (entry["n_frames"], 65, 65) or scalars.shape != (entry["n_frames"], 2):
                    raise ValueError(f"Invalid prediction shape: {entry}")
                if not np.isfinite(field).all() or not np.isfinite(scalars).all():
                    raise ValueError(f"Nonfinite predictions: {entry}")
                for name, array in (("psirz", field), ("q95", scalars[:, 0]), ("betaN", scalars[:, 1])):
                    with archive.open(f"shot_{i:04d}_{name}.npy", "w", force_zip64=True) as stream:
                        np.lib.format.write_array(stream, array.astype(np.float32), allow_pickle=False)
                frames += len(x)
            if (i + 1) % 100 == 0:
                print(f"Predicted {split}: {i + 1}/{len(entries)} shots", flush=True)
    temporary.replace(output)
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
    p.add_argument("--model", choices=["pod_deeponet", "pod_fourier_deeponet"], default="pod_deeponet")
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
    p.add_argument("--gradient-weight", type=float, default=0.25)
    p.add_argument("--gradient-region", choices=["auto", "all", "envelope"], default="auto",
                   help="Auto uses all-grid POD gradients and plasma-envelope Fourier gradients")
    p.add_argument("--field-gradient-weight", type=float, default=0.1)
    p.add_argument("--scalar-weight", type=float, default=0.25)
    p.add_argument("--ridge-alpha", type=float, default=-1.0,
                   help="Negative disables ridge warm start; optional nonnegative penalty fits the linear skips")
    p.add_argument("--li-dir", type=Path, help="Optional fixed training-contour internal-inductance targets")
    p.add_argument("--li-weight", type=float, default=0.02)
    p.add_argument("--li-sampled-only", action="store_true", help="Fine-tune only frames having valid training li supervision")
    p.add_argument("--geometry-weight", type=float, default=0.0,
                   help="Optional axis/LCFS displacement surrogate weight; requires geometry-augmented --li-dir")
    p.add_argument("--frame-weighting", choices=["none", "native_count"], default="none",
                   help="Optional training-only counts of original native indices nearest each retained frame")
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
    if args.frame_weighting != "none" and args.li_dir is not None:
        p.error("Native-count ablation currently weights the field/scalar losses and requires --li-dir to be omitted")
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
    Q = loss_matrix(tnorm, args.coefficient_power, args.gradient_weight, args.gradient_region)
    if args.gradient_region == "envelope":
        tnorm["field_gradient_weights"] = envelope_weights(device, torch.float32)
    li_store = li_module = geometry_module = None
    training_indices = torch.arange(len(training["X"]), device=device)
    if args.li_dir:
        from fusion.li_loss import LiTargetStore
        if args.geometry_weight:
            from fusion.geometry_loss import GeometryTargetStore
            li_store = GeometryTargetStore(args.li_dir)
            geometry_module = li_store.geometry_loss_module(device)
        else:
            li_store = LiTargetStore(args.li_dir)
        selected_hash = hashlib.sha256(json.dumps(split_entries(args.data_dir, "train"), sort_keys=True).encode()).hexdigest()
        if selected_hash != li_store.metadata["source_training_entries_sha256"]:
            raise ValueError("Internal-inductance target indexing differs from training entries")
        li_module = li_store.loss_module(device)
        if args.li_sampled_only:
            training_indices = torch.as_tensor(np.flatnonzero(li_store.lookup >= 0), device=device)
        print(f"Training li supervision: {len(np.flatnonzero(li_store.lookup >= 0))} native training frames", flush=True)
    train_psi = val_psi = None
    if args.model == "pod_fourier_deeponet":
        train_psi = torch.as_tensor(np.array(load_arrays(args.data_dir, "train", ("psi",))["psi"]), device=device)
        val_psi = torch.as_tensor(np.array(load_arrays(args.data_dir, "val", ("psi",))["psi"]), device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=args.lr * 0.03)
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
                                          *([ROOT / "fusion/frame_weights.py"] if frame_weighting else []),
                                          *([ROOT / "fusion/geometry_loss.py", ROOT / "fusion/li_loss.py"] if args.geometry_weight else []))}}
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
        if epoch:
            model.train()
            order = training_indices[torch.randperm(len(training_indices), device=device)]
            for indices in order.split(args.batch_size):
                batch = {key: value[indices] for key, value in training.items()}
                if ts_plans:
                    batch["X"] = drop_thomson_modalities(batch["X"], ts_plans, args.thomson_dropout)
                optimizer.zero_grad(set_to_none=True)
                li_positions, li_targets = li_store.batch(indices, device) if li_store else (None, None)
                loss = batch_loss(model, batch, tnorm, Q, args.scalar_weight,
                                  None if train_psi is None else train_psi[indices], args.field_gradient_weight,
                                  li_module=li_module, li_positions=li_positions, li_targets=li_targets, li_weight=args.li_weight,
                                  geometry_module=geometry_module, geometry_weight=args.geometry_weight)
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite training loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_loss += float(loss.detach()) * len(indices)
            scheduler.step()
        val_loss = validation(model, valid, tnorm, Q, args.scalar_weight, args.batch_size, val_psi, args.field_gradient_weight)
        record = {"epoch": epoch, "train_loss": train_loss / len(training_indices), "val_loss": val_loss,
                  "elapsed_seconds": elapsed_before + time.monotonic() - started, "lr": scheduler.get_last_lr()[0]}
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
