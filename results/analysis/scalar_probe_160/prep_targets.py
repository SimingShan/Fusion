#!/usr/bin/env python3
"""Targets and inputs for the direct-scalar-regression probe.

Question: can a dedicated regressor from the SAME 708 B-family inputs predict the seven consistency scalars
(R_axis, Z_axis, kappa, tri_top, tri_bot, volume, li) better on held-out shots than deriving them from the
blended psi? If yes, consistency is pipeline-limited (fixable); if not, it is input-limited.

Writes results/analysis/scalar_probe_160/{train,native,legacy}.npz with X (standardised, clipped +-12, 708),
Y (7, official derivation on the EFIT labels), valid mask, shot ids. Train Y: contour scalars from the official
geometry store contours (same official code), li from the store, axis from the official O-point finder on the label psi.
"""
import json, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__import__("os").environ.get("FUSION_ROOT") or Path(__file__).resolve().parents[3])
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "starter")); sys.path.insert(0, str(ROOT / "starter/fusion_scoring"))
from fusion.pod import load_arrays, read_manifest, standardize_x  # noqa: E402
import local_score as ls  # noqa: E402
from derive import derive_frame, _shape_from_contour, magnetic_axis  # noqa: E402

OUT = ROOT / "results/analysis/scalar_probe_160"; OUT.mkdir(parents=True, exist_ok=True)
NAMES = ["R_axis", "Z_axis", "kappa", "tri_top", "tri_bot", "volume", "li"]
norm = np.load(ROOT / "data/pod_history128_robust_v4/normalization.npz")
idx, mu, sd = norm["feature_indices"], norm["x_mean"].astype(np.float64), norm["x_std"].astype(np.float64)
mk = np.load(ROOT / "starter/fusion_scoring/masks/d3d_envelope.npz")
R, Z = mk["grid_R"], mk["grid_Z"]; mc = mk["mask_coarse"].astype(bool); mf = mc.astype(np.float64)
axs = ls.AXIS_SIGN["DIII-D"]
t0 = time.monotonic()


def standardize(X_full):
    """Exact model input path: feature selection + Thomson physical-scale transform + standardisation + clip."""
    return standardize_x(np.asarray(X_full), norm)


# ---- train: inputs from the pod view (already feature-selected), targets from store + label psi
Xtr = load_arrays(ROOT / "data/advanced_history_v2", "train", keys=("X",))["X"]  # raw (426684, 1348) incl. history sidecars
G = ROOT / "data/geometry_history_all_v5"
contours = np.load(G / "contours.npy", mmap_mode="r"); li = np.load(G / "li.npy"); valid = np.load(G / "valid.npy").astype(bool)
shot = np.load(G / "shot_indices.npy")
psi_tr = np.load(ROOT / "data/advanced_history_v2/consolidated/train_psi.npy", mmap_mode="r")
n = len(Xtr); Y = np.full((n, 7), np.nan)
for i in range(n):
    if valid[i]:
        s = _shape_from_contour(np.asarray(contours[i]))
        Y[i, 2:6] = [s["kappa"], s["tri_top"], s["tri_bot"], s["volume"]]
    Y[i, 6] = li[i]
    ra, za, _, _ = magnetic_axis(np.asarray(psi_tr[i], dtype=np.float64), R, Z, "DIII-D", mc, axs)
    Y[i, 0], Y[i, 1] = ra, za
    if i % 50000 == 0:
        print(f"[{time.monotonic()-t0:.0f}s] train {i}/{n}", flush=True)
ok = valid & np.isfinite(Y).all(1)
np.savez(OUT / "train.npz", X=standardize(Xtr), Y=Y.astype(np.float32), valid=ok, shot=shot)
print(f"train rows {n}, valid {ok.sum()}", flush=True)

# ---- dev folds: full official derivation on the label psi (truth) for every frame
for fold, ddir, split in (("native", "data/advanced_history_val_native_v3", "val"), ("legacy", "data/advanced_history_v2", "legacy_val")):
    d = ROOT / ddir
    arr = load_arrays(d, split, keys=("X", "psi"))
    X = np.asarray(arr["X"]); P = arr["psi"]
    Yd = np.full((len(X), 7), np.nan)
    for i in range(len(X)):
        v = derive_frame(np.asarray(P[i], dtype=np.float64), R, Z, "DIII-D", mc, mf, contour="auto", n_points=ls.N_POINTS, axis_sign=axs)
        Yd[i] = [v[k] for k in NAMES]
    ents = read_manifest(d)[split]
    sh = np.concatenate([np.full(e["n_frames"], e["shot_index"]) for e in ents])
    np.savez(OUT / f"{fold}.npz", X=standardize(X), Y=Yd.astype(np.float32), valid=np.isfinite(Yd).all(1), shot=sh)
    print(f"{fold}: {len(X)} frames, finite {np.isfinite(Yd).all(1).sum()} [{time.monotonic()-t0:.0f}s]", flush=True)
print("done", flush=True)
