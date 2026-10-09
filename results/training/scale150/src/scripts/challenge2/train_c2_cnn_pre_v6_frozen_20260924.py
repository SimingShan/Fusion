#!/usr/bin/env python3
"""Train the machine-agnostic Challenge-2 flux U-Net on DIII-D and select by the official scorer.

Target: phi = -efit_psirz (axis maximum); prediction phi_hat = phi_ext + F * (resid + offset)
with F = mu0 |Ip| L / 2. Loss (feature-spec v2):
    MSE(resid field) / var_train(target) + gradient_weight * MSE(finite-difference gradients) / norm_grad
    + scalar_weight * MSE(z-scored q95/betaN, NaN-masked)
where var_train(target) is the variance of the training-set residual (phi - phi_ext)/F over all kept frames and
pixels, computed once at startup (--field-norm auto), so the field term starts near 1 like the scalar term;
norm_grad is the same variance by default (--grad-norm field, keeps the v1 field:gradient balance) or the
variance of the target's finite differences (--grad-norm gradient). Defaults: gradient_weight 0.25,
scalar_weight 0.1. The model takes a field-conditioning vector (FiLM) and a separate scalar-head
conditioning vector (fusion.c2_fields_data.FIELD_COND_NAMES / SCALAR_COND_NAMES); the feature spec is
written to run.json and every checkpoint and a differing spec is refused on load (--predict-only, --init).
Every --score-every epochs the native dev fold is predicted in the official npz layout and scored with
scripts/evaluate.py; the best checkpoint by official S is kept (best.pt). --predict-only writes val/test
predictions from best.pt.

Synthetic MAST augmentation (v4 training path, feature spec unchanged): --synthetic-dir points at shards written by
scripts/challenge2/generate_synthetic_mast.py (free-boundary GS equilibria on REAL MAST public inputs, fusion/c2_synthetic.py).
Each batch draws a fraction --synthetic-weight of its frames from them; they are featurised with the MAST MachineSetup
(MAST coil tables, grid, envelope, L) exactly as predict_mast_cnn.py featurises real MAST inputs, with target
phi = +psi (axis maximum) and the same F normalisation. q95/betaN are z-scored with the DIII-D training statistics
for both machines (one machine-agnostic head, one inverse at inference; the synthetic MAST q95 3-9 / betaN 1-4 lie
within ~2 sigma of the DIII-D statistics), so predict_mast_cnn.py needs no change. Checkpoint selection stays the
official DIII-D dev-fold S; a held-out synthetic MAST split (by source shot) is additionally scored with the official
scorer machinery (fusion.c2_synthetic.score_synthetic) at every scoring epoch and the best such checkpoint is kept as
best_synthetic.pt.

Label use: DIII-D train/val caches only. Never touches audit splits or MAST labels.
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from fusion import c2_fields_data as cd  # noqa: E402
from fusion.c2_unet import C2UNet, count_parameters  # noqa: E402

FORBIDDEN_SPLITS = ("audit",)


def uniform_indices(n, cap):
    return np.arange(n) if not cap or cap >= n else np.linspace(0, n - 1, cap, dtype=int)


def source_hashes():
    files = [ROOT / "fusion/c2_fields_data.py", ROOT / "fusion/c2_unet.py", ROOT / "fusion/physics_fields.py",
             Path(__file__).resolve()]
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def batch_loss(model, feats, scalar_norm, loss_norm, gradient_weight, scalar_weight, return_parts=False):
    """Normalised field/gradient MSE plus z-scored scalar MSE; parts are reported both raw and normalised.

    With return_parts the per-sample field MSE (B,) is included ("field_mse_samples") so a mixed DIII-D/synthetic
    batch can be logged per source."""
    pred = model(feats["channels"], feats["cond"], feats["scalar_cond"])
    resid = pred["field"] + pred["offset"][:, None, None]
    error = resid - feats["target"]
    field_samples = error.square().flatten(1).mean(1)
    field_mse = field_samples.mean()
    gr = error[:, :, 1:] - error[:, :, :-1]
    gz = error[:, 1:] - error[:, :-1]
    grad_mse = gr.square().flatten(1).mean(1).mean() + gz.square().flatten(1).mean(1).mean()
    target_z = (feats["scalars"] - scalar_norm["mean"]) / scalar_norm["std"]
    finite = torch.isfinite(target_z)
    scalar_error = (pred["scalars"] - torch.nan_to_num(target_z, nan=0.0)).square() * finite
    scalar_loss = scalar_error.sum() / finite.sum().clamp_min(1)
    field_loss = field_mse / loss_norm["field"]
    grad_loss = grad_mse / loss_norm["grad"]
    loss = field_loss + gradient_weight * grad_loss + scalar_weight * scalar_loss
    if return_parts:
        return loss, {"field": float(field_loss), "grad": float(grad_loss), "scalar": float(scalar_loss),
                      "field_mse_raw": float(field_mse), "grad_mse_raw": float(grad_mse),
                      "field_weighted": float(field_loss), "grad_weighted": gradient_weight * float(grad_loss),
                      "scalar_weighted": scalar_weight * float(scalar_loss), "field_mse_samples": field_samples.detach()}
    return loss


@torch.no_grad()
def target_statistics(featurizer, store_t, indices, batch_size):
    """Training-set variance of the residual target (phi - phi_ext)/F and of its finite differences (one pass)."""
    n = s1 = s2 = 0.0
    gn = g1 = g2 = 0.0
    for first in range(0, len(indices), batch_size):
        feats = featurizer(cd.batch_from_store(store_t, indices[first:first + batch_size]))
        t = feats["target"].double()
        n += t.numel(); s1 += float(t.sum()); s2 += float(t.square().sum())
        for g in (t[:, :, 1:] - t[:, :, :-1], t[:, 1:] - t[:, :-1]):
            gn += g.numel(); g1 += float(g.sum()); g2 += float(g.square().sum())
    var = s2 / n - (s1 / n) ** 2
    gvar = 2.0 * (g2 / gn - (g1 / gn) ** 2)   # the gradient term sums the R and Z mean squares
    return {"target_mean": s1 / n, "target_var": var, "target_second_moment": s2 / n,
            "gradient_var_sum": gvar, "pixels": n}


@torch.no_grad()
def predict_store(model, featurizer, store_t, n, batch_size, scalar_norm, device):
    """phi_hat (n,65,65) float32 numpy and scalars (n,2) for a whole torch store."""
    model.eval()
    fields, scalars = [], []
    for first in range(0, n, batch_size):
        idx = torch.arange(first, min(first + batch_size, n), device=device)
        feats = featurizer(cd.batch_from_store(store_t, idx))
        pred = model(feats["channels"], feats["cond"], feats["scalar_cond"])
        phi_hat = feats["phi_ext"] + feats["F"][:, None, None] * (pred["field"] + pred["offset"][:, None, None])
        fields.append(phi_hat.float().cpu().numpy())
        scalars.append((pred["scalars"] * scalar_norm["std"] + scalar_norm["mean"]).float().cpu().numpy())
    return np.concatenate(fields), np.concatenate(scalars)


def write_official_npz(output: Path, arrays: dict):
    """Official key layout shot_%04d_{psirz,q95,betaN}; float32 members, streamed into the zip."""
    temporary = output.with_suffix(".tmp.npz")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as archive:
        for key, value in arrays.items():
            if not np.isfinite(value).all():
                raise ValueError(f"Nonfinite prediction {key}")
            with archive.open(f"{key}.npy", "w", force_zip64=True) as stream:
                np.lib.format.write_array(stream, np.ascontiguousarray(value.astype(np.float32)), allow_pickle=False)
    temporary.replace(output)


def predict_d3d_split(model, featurizer, index, data_dir: Path, split: str, positions, output: Path,
                      batch_size, scalar_norm, device, axis_sign=-1.0):
    """Predict selected manifest positions of a prepared DIII-D split; psi = axis_sign * phi_hat."""
    entries = cd.split_entries(data_dir, split)
    arrays = {}
    frames = 0
    started = time.monotonic()
    for count, i in enumerate(positions):
        shot = cd.d3d_shot_inputs(data_dir / entries[i]["path"], index, 0, with_labels=False)
        store = cd.FrameStore([shot], K=featurizer_K(featurizer), with_labels=False)
        phi_hat, scalars = predict_store(model, featurizer, store.to_torch(device, False), store.n, batch_size, scalar_norm, device)
        if phi_hat.shape != (entries[i]["n_frames"], 65, 65):
            raise ValueError(f"{entries[i]['path']}: predicted {phi_hat.shape}, cache has {entries[i]['n_frames']} frames")
        arrays[f"shot_{i:04d}_psirz"] = axis_sign * phi_hat
        arrays[f"shot_{i:04d}_q95"] = scalars[:, 0]
        arrays[f"shot_{i:04d}_betaN"] = scalars[:, 1]
        frames += store.n
        if (count + 1) % 50 == 0:
            print(f"predicted {split}: {count + 1}/{len(positions)} shots", flush=True)
    write_official_npz(output, arrays)
    print(f"wrote {output}: {len(positions)} shots, {frames} frames in {time.monotonic() - started:.1f} s", flush=True)
    return frames


def featurizer_K(featurizer):
    return getattr(featurizer, "K", None)


MERGE_KEYS = ("channels", "cond", "scalar_cond", "target", "scalars")


def merge_features(*feats):
    """Concatenate featurizer outputs of different machines along the batch axis (all channels are dimensionless)."""
    return {k: torch.cat([f[k] for f in feats], dim=0) for k in MERGE_KEYS}


def load_synthetic(args, setup_mast, device, verbose=True):
    """Synthetic MAST train store (torch) + held-out synthetic val shots/store for the official-scorer evaluation."""
    from fusion import c2_synthetic as cs
    train_shots = cs.load_synthetic_shots(args.synthetic_dir, "train", setup_mast.table_names,
                                          diverted_only=args.synthetic_diverted_only)
    val_shots = cs.load_synthetic_shots(args.synthetic_dir, "val", setup_mast.table_names, max_frames=args.synthetic_val_frames,
                                        diverted_only=args.synthetic_diverted_only, seed=args.seed)
    train_store = cs.synthetic_store(train_shots)
    val_store = cs.synthetic_store(val_shots, K=train_store.K)
    info = {"dir": str(args.synthetic_dir), "weight": args.synthetic_weight, "train_frames": int(train_store.n),
            "train_source_shots": len(train_shots), "val_frames": int(val_store.n), "val_source_shots": len(val_shots),
            "diverted_only": bool(args.synthetic_diverted_only), "K": int(train_store.K),
            "manifest": cd.read_manifest(args.synthetic_dir).get("arguments")}
    if verbose:
        print(json.dumps({"synthetic": {k: v for k, v in info.items() if k != "manifest"}}), flush=True)
    return train_store, val_shots, val_store, info


@torch.no_grad()
def evaluate_synthetic(model, featurizer_mast, val_shots, val_store_t, n, batch_size, scalar_norm, device, workers, output: Path,
                       reference_cache: Path | None):
    """Predict the held-out synthetic MAST frames and score them with the official scorer machinery (MAST envelope)."""
    from fusion import c2_synthetic as cs
    started = time.monotonic()
    phi_hat, scalars = predict_store(model, featurizer_mast, val_store_t, n, batch_size, scalar_norm, device)
    metrics = cs.score_synthetic(val_shots, phi_hat.astype(np.float64), scalars.astype(np.float64), workers=workers,
                                 reference_cache=reference_cache)
    metrics["eval_seconds"] = time.monotonic() - started
    output.write_text(json.dumps(metrics, indent=2, default=float) + "\n")
    print(f"synthetic MAST val S={metrics['S']:.6f} r2_psi={metrics['r2_psi']:.6f} r2_qb={metrics.get('r2_qb')} "
          f"dlcfs={metrics.get('dlcfs')} cons={metrics.get('consistency')} ({n} frames, {metrics['eval_seconds']:.0f} s)", flush=True)
    return metrics


def score_predictions(data_dir: Path, split: str, pred: Path, output: Path, workers: int, max_shots: int) -> dict:
    cmd = [sys.executable, str(ROOT / "scripts/evaluate.py"), "--data-dir", str(data_dir), "--split", split,
           "--pred", str(pred), "--output", str(output), "--workers", str(workers), "--max-shots", str(max_shots)]
    with output.with_suffix(".log").open("w") as log:
        subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
    metrics = json.loads(output.read_text())
    print(f"official local {split} S={metrics['S']:.6f} r2_psi={metrics['r2_psi']:.6f} "
          f"r2_qb={metrics.get('r2_qb')} dlcfs={metrics.get('dlcfs')} cons={metrics.get('consistency')}", flush=True)
    return metrics


def build_model(kwargs, device):
    return C2UNet(**kwargs).to(device)


def load_checkpoint(path: Path, device):
    """Load a checkpoint; refuses one whose feature spec (channels/conditioning contract) differs from the code."""
    saved = torch.load(path, map_location="cpu", weights_only=False)
    cd.check_feature_spec(saved.get("feature_spec"), what=str(path))
    model = build_model(saved["model_kwargs"], device)
    model.load_state_dict(saved["state_dict"])
    scalar_norm = {k: torch.as_tensor(v, device=device, dtype=torch.float32) for k, v in saved["scalar_norm"].items()}
    return model, scalar_norm, saved


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=Path("data/advanced_v1"), help="prepared DIII-D train caches")
    p.add_argument("--val-dir", type=Path, default=Path("data/advanced_val_native_v3"), help="native dev fold caches")
    p.add_argument("--val-split", default="val")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--frames-per-shot", type=int, default=32)
    p.add_argument("--max-train-shots", type=int, default=0)
    p.add_argument("--train-shot-list", type=Path, help="JSON list of manifest shot_index values to train on")
    p.add_argument("--min-ip-kA", type=float, default=100.0, help="drop training frames with |Ip| below this")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--seconds", type=int, default=25000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--gradient-weight", type=float, default=0.25, help="weight of the normalised gradient MSE")
    p.add_argument("--scalar-weight", type=float, default=0.1, help="weight of the z-scored q95/betaN MSE")
    p.add_argument("--field-norm", default="auto",
                   help="divisor of the field MSE: 'auto' = training-set variance of (phi-phi_ext)/F, 'none' = 1, or a float")
    p.add_argument("--grad-norm", choices=["field", "gradient"], default="field",
                   help="divisor of the gradient MSE: the field divisor (v1 balance) or the target finite-difference variance")
    p.add_argument("--base", type=int, default=32)
    p.add_argument("--levels", type=int, default=3)
    p.add_argument("--score-every", type=int, default=5)
    p.add_argument("--val-max-shots", type=int, default=0, help="evaluate.py --max-shots (0 = full fold)")
    p.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--init", type=Path)
    p.add_argument("--predict-only", action="store_true")
    p.add_argument("--split", choices=["val", "legacy_val", "test"], default="val")
    p.add_argument("--pred-name", default=None, help="override output npz filename for --predict-only")
    p.add_argument("--no-score", action="store_true")
    p.add_argument("--synthetic-dir", type=Path, default=None,
                   help="synthetic MAST equilibria shards (generate_synthetic_mast.py); enables mixed-machine training")
    p.add_argument("--synthetic-weight", type=float, default=0.3, help="fraction of every batch drawn from synthetic MAST frames")
    p.add_argument("--synthetic-val-frames", type=int, default=800, help="held-out synthetic frames scored per evaluation (0 = all)")
    p.add_argument("--synthetic-diverted-only", action="store_true", help="drop envelope-limited synthetic equilibria")
    args = p.parse_args()
    if args.synthetic_dir is not None and not (0.0 < args.synthetic_weight < 1.0):
        p.error("--synthetic-weight must lie strictly between 0 and 1")
    if args.val_split in FORBIDDEN_SPLITS or args.split in FORBIDDEN_SPLITS:
        p.error("audit splits are never read")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.workers)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = True
    setup = cd.build_machine_setup("DIII-D")
    featurizer = cd.FrameFeaturizer(setup, device)
    featurizer.K = cd.D3D_K
    feature_names = cd.read_manifest(args.data_dir)["feature_names"]
    index = cd.D3DFeatureIndex(feature_names, setup.table_names)

    if args.predict_only:
        model, scalar_norm, saved = load_checkpoint(args.output / "best.pt", device)
        if saved.get("feature_names") not in (None, feature_names):
            raise ValueError("Prediction feature order differs from the training contract")
        print(f"loaded {args.output / 'best.pt'} (epoch {saved.get('epoch')}, feature spec {cd.feature_spec_hash()})", flush=True)
        if args.split == "test":
            data_dir, split, filename = args.data_dir, "test", "diii_d_public_test.npz"
        else:
            data_dir, split, filename = args.val_dir, args.split, f"{args.split}_predictions.npz"
        if cd.read_manifest(data_dir)["feature_names"] != feature_names:
            raise ValueError(f"{data_dir}: feature names differ")
        filename = args.pred_name or filename
        entries = cd.split_entries(data_dir, split)
        positions = uniform_indices(len(entries), args.val_max_shots).tolist()
        pred_path = args.output / filename
        predict_d3d_split(model, featurizer, index, data_dir, split, positions, pred_path, args.batch_size, scalar_norm, device)
        if split != "test" and not args.no_score:
            score_predictions(data_dir, split, pred_path, pred_path.with_name(Path(filename).stem + "_score.json"),
                              args.workers, args.val_max_shots)
        return

    started = time.monotonic()
    shot_indices = None
    if args.train_shot_list:
        shot_indices = set(int(v) for v in json.loads(args.train_shot_list.read_text()))
    shots, positions = cd.load_d3d_split(args.data_dir, "train", index, args.frames_per_shot, args.max_train_shots, shot_indices)
    store = cd.FrameStore(shots, K=cd.D3D_K)
    keep = store.frame_mask(args.min_ip_kA)
    train_idx = torch.as_tensor(np.flatnonzero(keep), device=device)
    finite = np.isfinite(store.scalars).all(1)
    scalar_mean = np.nanmean(store.scalars[keep & finite], axis=0)
    scalar_std = np.nanstd(store.scalars[keep & finite], axis=0) + 1e-6
    scalar_norm = {"mean": torch.as_tensor(scalar_mean, device=device, dtype=torch.float32),
                   "std": torch.as_tensor(scalar_std, device=device, dtype=torch.float32)}
    store_t = store.to_torch(device)
    target_stats = target_statistics(featurizer, store_t, train_idx, args.batch_size)
    if args.field_norm == "auto":
        field_norm = target_stats["target_var"]
    elif args.field_norm == "none":
        field_norm = 1.0
    else:
        field_norm = float(args.field_norm)
    if not field_norm > 0:
        raise ValueError(f"field normaliser must be positive, got {field_norm}")
    grad_norm = field_norm if args.grad_norm == "field" else target_stats["gradient_var_sum"]
    loss_norm = {"field": field_norm, "grad": grad_norm}
    loss_normalisation = {"field_divisor": field_norm, "grad_divisor": grad_norm, "field_norm_mode": args.field_norm,
                          "grad_norm_mode": args.grad_norm, "gradient_weight": args.gradient_weight,
                          "scalar_weight": args.scalar_weight, "target_statistics": target_stats}
    synthetic = None
    if args.synthetic_dir is not None:
        setup_mast = cd.build_machine_setup("MAST")
        featurizer_mast = cd.FrameFeaturizer(setup_mast, device)
        syn_store, syn_val_shots, syn_val_store, syn_info = load_synthetic(args, setup_mast, device)
        syn_store_t = syn_store.to_torch(device)
        syn_val_t = syn_val_store.to_torch(device)
        syn_stats = target_statistics(featurizer_mast, syn_store_t, torch.arange(syn_store.n, device=device), args.batch_size)
        syn_info["target_statistics"] = syn_stats
        syn_info["scalar_norm"] = "DIII-D training statistics (shared machine-agnostic head; see module docstring)"
        syn_info["mast_setup"] = setup_mast.summary()
        syn_scalar_mean = np.nanmean(syn_store.scalars, axis=0).tolist()
        syn_info["synthetic_scalar_mean"] = syn_scalar_mean
        syn_info["synthetic_scalar_z_of_mean"] = ((np.asarray(syn_scalar_mean) - scalar_mean) / scalar_std).tolist()
        n_syn_batch = max(1, int(round(args.synthetic_weight * args.batch_size)))
        n_d3d_batch = max(1, args.batch_size - n_syn_batch)
        syn_info["frames_per_batch"] = {"synthetic": n_syn_batch, "diii_d": n_d3d_batch}
        synthetic = {"store_t": syn_store_t, "n": syn_store.n, "val_shots": syn_val_shots, "val_t": syn_val_t, "val_n": syn_val_store.n,
                     "featurizer": featurizer_mast, "n_batch": n_syn_batch, "info": syn_info}
        print(json.dumps({"synthetic_target_statistics": syn_stats, "synthetic_scalar_z_of_mean": syn_info["synthetic_scalar_z_of_mean"],
                          "frames_per_batch": syn_info["frames_per_batch"]}), flush=True)
    else:
        n_d3d_batch = args.batch_size
    load_seconds = time.monotonic() - started
    print(json.dumps({"train_shots": len(shots), "train_frames": int(store.n), "kept_frames": int(keep.sum()),
                      "dropped_low_ip": int((~keep).sum()), "load_seconds": load_seconds,
                      "scalar_mean": scalar_mean.tolist(), "scalar_std": scalar_std.tolist(), "device": str(device),
                      "loss_normalisation": loss_normalisation}), flush=True)

    kwargs = dict(in_channels=cd.N_CHANNELS, cond_dim=cd.N_FIELD_COND, scalar_cond_dim=cd.N_SCALAR_COND,
                  base=args.base, levels=args.levels)
    model = build_model(kwargs, device)
    if args.init:
        saved = torch.load(args.init, map_location="cpu", weights_only=False)
        cd.check_feature_spec(saved.get("feature_spec"), what=str(args.init))
        model.load_state_dict(saved["state_dict"])
        print(f"warm started from {args.init}", flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=args.lr * 0.03)
    val_entries = cd.split_entries(args.val_dir, args.val_split)
    val_positions = uniform_indices(len(val_entries), args.val_max_shots).tolist()
    run = {"model": "c2_unet", "model_kwargs": kwargs, "parameters": count_parameters(model),
           "feature_spec": cd.feature_spec(), "feature_spec_hash": cd.feature_spec_hash(),
           "channel_names": cd.CHANNEL_NAMES, "field_cond_names": cd.FIELD_COND_NAMES,
           "scalar_cond_names": cd.SCALAR_COND_NAMES, "loss_normalisation": loss_normalisation,
           "machine_setup": setup.summary(),
           "train_shots": len(shots), "train_frames": int(store.n), "training_frames_per_epoch": int(keep.sum()),
           "val_shots_scored": len(val_positions), "val_frames_available": int(sum(e["n_frames"] for e in val_entries)),
           "device": str(device), "gpu": torch.cuda.get_device_name() if device.type == "cuda" else None,
           "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
           "checkpoint_selection": "maximum official composite S on the native dev fold at scheduled epochs",
           "started_utc": datetime.now(timezone.utc).isoformat(), "audit_labels_read": False, "mast_labels_read": False,
           "scalar_norm": {"mean": scalar_mean.tolist(), "std": scalar_std.tolist()},
           "synthetic": None if synthetic is None else synthetic["info"],
           "source_sha256": source_hashes(), "status": "running"}
    (args.output / "run.json").write_text(json.dumps(run, indent=2) + "\n")
    print(json.dumps({k: v for k, v in run.items() if k not in ("machine_setup", "source_sha256", "loss_normalisation", "synthetic")}), flush=True)

    history, best_score, best_synthetic = [], -float("inf"), -float("inf")
    n_train = len(train_idx)
    syn_order, syn_cursor = None, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_started = time.monotonic()
        total = 0.0
        parts_sum = {"field": 0.0, "grad": 0.0, "scalar": 0.0, "field_mse_raw": 0.0, "grad_mse_raw": 0.0,
                     "field_weighted": 0.0, "grad_weighted": 0.0, "scalar_weighted": 0.0}
        source_mse = {"diii_d": 0.0, "synthetic": 0.0}
        source_count = {"diii_d": 0, "synthetic": 0}
        order = train_idx[torch.randperm(n_train, device=device)]
        n_seen = 0
        for indices in order.split(n_d3d_batch):
            feats = featurizer(cd.batch_from_store(store_t, indices))
            n_d = len(indices)
            if synthetic is not None:
                if syn_order is None or syn_cursor + synthetic["n_batch"] > len(syn_order):
                    syn_order, syn_cursor = torch.randperm(synthetic["n"], device=device), 0
                syn_idx = syn_order[syn_cursor:syn_cursor + synthetic["n_batch"]]
                syn_cursor += synthetic["n_batch"]
                feats = merge_features(feats, synthetic["featurizer"](cd.batch_from_store(synthetic["store_t"], syn_idx)))
            n_batch = feats["channels"].shape[0]
            optimizer.zero_grad(set_to_none=True)
            loss, parts = batch_loss(model, feats, scalar_norm, loss_norm, args.gradient_weight, args.scalar_weight,
                                     return_parts=True)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach()) * n_batch
            n_seen += n_batch
            for k in parts_sum:
                parts_sum[k] += parts[k] * n_batch
            samples = parts["field_mse_samples"]
            source_mse["diii_d"] += float(samples[:n_d].sum()); source_count["diii_d"] += n_d
            if n_batch > n_d:
                source_mse["synthetic"] += float(samples[n_d:].sum()); source_count["synthetic"] += n_batch - n_d
        scheduler.step()
        record = {"epoch": epoch, "train_loss": total / n_seen, **{f"train_{k}": v / n_seen for k, v in parts_sum.items()},
                  "train_field_mse_raw_diii_d": source_mse["diii_d"] / max(source_count["diii_d"], 1),
                  "train_field_mse_raw_synthetic": (source_mse["synthetic"] / source_count["synthetic"]) if source_count["synthetic"] else None,
                  "frames_seen": {"diii_d": source_count["diii_d"], "synthetic": source_count["synthetic"]},
                  "epoch_seconds": time.monotonic() - epoch_started, "elapsed_seconds": time.monotonic() - started,
                  "lr": scheduler.get_last_lr()[0]}
        checkpoint = dict(model="c2_unet", model_kwargs=kwargs, state_dict=model.state_dict(), epoch=epoch,
                          scalar_norm={"mean": scalar_mean.tolist(), "std": scalar_std.tolist()},
                          feature_names=feature_names, feature_spec=cd.feature_spec(), channel_names=cd.CHANNEL_NAMES,
                          field_cond_names=cd.FIELD_COND_NAMES, scalar_cond_names=cd.SCALAR_COND_NAMES,
                          loss_normalisation=loss_normalisation, machine_setup=setup.summary(), run=run,
                          synthetic=None if synthetic is None else {k: v for k, v in synthetic["info"].items() if k != "manifest"})
        torch.save(checkpoint, args.output / "last.pt")
        final_epoch = epoch == args.epochs or record["elapsed_seconds"] >= args.seconds
        if epoch % args.score_every == 0 or final_epoch:
            pred_path = args.output / "candidate_val_predictions.npz"
            score_path = args.output / "candidate_val_score.json"
            eval_started = time.monotonic()
            predict_d3d_split(model, featurizer, index, args.val_dir, args.val_split, val_positions, pred_path,
                              args.batch_size, scalar_norm, device)
            metrics = score_predictions(args.val_dir, args.val_split, pred_path, score_path, args.workers, args.val_max_shots)
            record.update(S=metrics["S"], r2_psi=metrics["r2_psi"], r2_qb=metrics.get("r2_qb"), dlcfs=metrics.get("dlcfs"),
                          consistency=metrics.get("consistency"), eval_seconds=time.monotonic() - eval_started)
            checkpoint["validation_score"] = metrics["S"]
            if synthetic is not None:
                syn_metrics = evaluate_synthetic(model, synthetic["featurizer"], synthetic["val_shots"], synthetic["val_t"], synthetic["val_n"],
                                                 args.batch_size, scalar_norm, device, args.workers,
                                                 args.output / "candidate_synthetic_val_score.json",
                                                 args.output / "synthetic_val_reference.npz")
                record.update(synthetic_S=syn_metrics["S"], synthetic_r2_psi=syn_metrics["r2_psi"], synthetic_r2_qb=syn_metrics.get("r2_qb"),
                              synthetic_dlcfs=syn_metrics.get("dlcfs"), synthetic_consistency=syn_metrics.get("consistency"),
                              synthetic_r2_cons_each=syn_metrics.get("r2_cons_each"), synthetic_eval_seconds=syn_metrics["eval_seconds"])
                checkpoint["synthetic_validation_score"] = syn_metrics["S"]
                if syn_metrics["S"] > best_synthetic:
                    best_synthetic = syn_metrics["S"]
                    torch.save(checkpoint, args.output / "best_synthetic.pt")
                    (args.output / "candidate_synthetic_val_score.json").replace(args.output / "synthetic_val_score.json")
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
        print(json.dumps(record), flush=True)
        if final_epoch:
            break
    run.update(completed_epochs=epoch, elapsed_seconds=time.monotonic() - started, best_validation_score=best_score,
               best_synthetic_validation_score=(best_synthetic if synthetic is not None else None), status="completed")
    (args.output / "run.json").write_text(json.dumps(run, indent=2) + "\n")
    print(f"completed; best official local validation S={best_score:.6f}"
          + (f"; best synthetic MAST S={best_synthetic:.6f}" if synthetic is not None else ""), flush=True)


if __name__ == "__main__":
    main()
