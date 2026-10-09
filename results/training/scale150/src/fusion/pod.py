"""Train-only POD preprocessing and memory-mapped operator training data."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def read_manifest(data_dir):
    return json.loads((Path(data_dir) / "manifest.json").read_text())


def split_entries(data_dir, split):
    manifest = read_manifest(data_dir)
    return manifest.get("splits", manifest)[split]


def validate_splits(data_dir):
    manifest = read_manifest(data_dir)
    splits = manifest.get("splits", manifest)
    seen = set()
    for name in ("train", "val", "audit", "legacy_val"):
        entries = splits.get(name, [])
        ids = [e["shot_index"] for e in entries]
        if len(set(ids)) != len(ids) or seen.intersection(ids):
            raise ValueError(f"Duplicate or overlapping labeled shots in {name}")
        seen.update(ids)
    if not splits["train"] or not splits["val"]:
        raise ValueError("Nonempty training and validation splits are required")


def load_arrays(data_dir, split, keys=("X", "psi", "scalars")):
    data_dir = Path(data_dir)
    paths = {key: data_dir / "consolidated" / f"{split}_{key}.npy" for key in keys}
    if all(path.exists() for path in paths.values()):
        arrays = {key: np.load(path, mmap_mode="r", allow_pickle=False) for key, path in paths.items()}
    else:
        parts = {key: [] for key in keys}
        for entry in split_entries(data_dir, split):
            with np.load(data_dir / entry["path"], allow_pickle=False) as shot:
                for key in keys:
                    values = shot[key]
                    if key == "X" and entry.get("X_extra"):
                        extra = np.load(data_dir / entry["X_extra"], allow_pickle=False)
                        if len(extra) != len(values):
                            raise ValueError("History sidecar frame count differs from base features")
                        values = np.concatenate((values, extra), axis=1)
                    parts[key].append(values)
        arrays = {key: np.concatenate(value) for key, value in parts.items()}
    expected = sum(e["n_frames"] for e in split_entries(data_dir, split))
    if any(len(value) != expected for value in arrays.values()):
        raise ValueError(f"Consolidated arrays disagree with {split} manifest")
    return arrays


def manifest_hash(data_dir):
    return hashlib.sha256((Path(data_dir) / "manifest.json").read_bytes()).hexdigest()


def standardize_x(X, norm):
    selected = np.asarray(X[:, norm["feature_indices"]], dtype=np.float32)
    if "raw_log_scale" in norm:
        selected = selected.astype(np.float64)
        scales = norm["raw_log_scale"][norm["feature_indices"]]
        chosen = np.flatnonzero(scales != 1.0)
        if len(chosen):
            selected[:, chosen] = np.log1p(np.expm1(selected[:, chosen]) / scales[chosen])
    return np.clip((selected - norm["x_mean"]) / norm["x_std"], -12, 12).astype(np.float32)


def fit_pod(data_dir, output, rank=128, basis_samples=16384, seed=2026, feature_limit=0):
    """Fit the mean and basis only on training frames; never inspect audit labels."""
    from scipy.linalg import eigh
    import time

    started = time.monotonic()
    data_dir, output = Path(data_dir), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    validate_splits(data_dir)
    train = load_arrays(data_dir, "train")
    X = train["X"][:, :feature_limit or None]
    x_mean = X.mean(axis=0, dtype=np.float64)
    x_std = X.std(axis=0, dtype=np.float64)
    active = np.flatnonzero(x_std > 1e-6)
    flat = train["psi"].reshape(len(X), -1)
    if rank > min(flat.shape):
        raise ValueError("POD rank exceeds available training dimensions")
    mean = flat.mean(axis=0, dtype=np.float64)
    sumsquares = 0.0
    for first in range(0, len(flat), 2048):
        values = np.asarray(flat[first:first + 2048], dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite training flux")
        sumsquares += np.square(values).sum()
    psi_std = np.sqrt(max(sumsquares / flat.size - mean.mean() ** 2, 1e-12))
    chosen = np.sort(np.random.default_rng(seed).choice(len(flat), min(basis_samples, len(flat)), replace=False))
    sample = np.asarray(flat[chosen], dtype=np.float64) - mean
    print(f"POD covariance: {sample.shape}, rank {rank}", flush=True)
    covariance = sample.T @ sample / len(sample)
    del sample
    eigenvalues, eigenvectors = eigh(covariance, subset_by_index=(flat.shape[1] - rank, flat.shape[1] - 1), driver="evr", check_finite=False)
    basis = np.ascontiguousarray(eigenvectors[:, ::-1].T, dtype=np.float32)
    eigenvalues = np.maximum(eigenvalues[::-1], 0)
    scalars = np.asarray(train["scalars"], dtype=np.float64)
    scalars[~np.isfinite(scalars)] = np.nan
    norm = dict(feature_indices=active, x_mean=x_mean[active].astype(np.float32),
                x_std=x_std[active].astype(np.float32), psi_mean=mean.astype(np.float32).reshape(65, 65),
                psi_std=np.float32(psi_std), basis=basis,
                coefficient_std=np.maximum(np.sqrt(eigenvalues), 1e-6).astype(np.float32),
                eigenvalues=eigenvalues, scalar_mean=np.nanmean(scalars, axis=0).astype(np.float32),
                scalar_std=np.maximum(np.nanstd(scalars, axis=0), 1e-6).astype(np.float32))
    if not all(np.isfinite(a).all() for a in norm.values()):
        raise ValueError("Nonfinite POD normalization")
    np.savez(output / "normalization.npz", **norm)
    metadata = {"source_manifest_sha256": manifest_hash(data_dir), "data_dir": str(data_dir.resolve()),
                "seed": seed, "rank": rank, "basis_samples": len(chosen), "basis_sample_indices_sha256": hashlib.sha256(chosen.tobytes()).hexdigest(),
                "basis_fit_split": "train", "training_frames": len(X), "input_features": X.shape[1],
                "active_features": len(active), "feature_limit": feature_limit,
                "basis_variance_fraction": float(eigenvalues.sum() / np.trace(covariance)),
                "train_shots": len(split_entries(data_dir, "train")), "audit_labels_read": False,
                "seconds": time.monotonic() - started}
    (output / "basis.json").write_text(json.dumps(metadata, indent=2) + "\n")
    np.save(output / "basis_sample_indices.npy", chosen)
    print(json.dumps(metadata), flush=True)
    return norm


def project_split(data_dir, output, split, norm):
    """Prepare compact targets for efficient training. Audit is opt-in."""
    output = Path(output)
    arrays = load_arrays(data_dir, split)
    n = len(arrays["X"])
    projected = {}
    shapes = {"X": (n, len(norm["feature_indices"])), "coefficients": (n, len(norm["basis"])), "scalars": (n, 2)}
    for name, shape in shapes.items():
        projected[name] = np.lib.format.open_memmap(output / f"{split}_{name}.npy", mode="w+", dtype="float32", shape=shape)
    for first in range(0, n, 1024):
        sl = slice(first, first + 1024)
        projected["X"][sl] = standardize_x(arrays["X"][sl], norm)
        centered = arrays["psi"][sl].reshape(-1, 4225) - norm["psi_mean"].reshape(-1)
        projected["coefficients"][sl] = (centered @ norm["basis"].T) / norm["coefficient_std"]
        projected["scalars"][sl] = (arrays["scalars"][sl] - norm["scalar_mean"]) / norm["scalar_std"]
    for value in projected.values():
        value.flush()
    print(f"Projected {split}: {n} frames", flush=True)
