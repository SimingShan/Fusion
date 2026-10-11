#!/usr/bin/env python3
"""9-scalar copy of the all-labelled LVF view for the HLVF members: data/pod_history128_alllab_vf_heads_v1 = data/pod_history128_alllab_vf_v1
(every array symlinked) except *_scalars.npy = [q95, betaN (as stored), standardised consistency scalars] and normalization.npz (scalar_mean /
scalar_std extended). Rows = training rows (results/analysis/scalar_probe_160/train.npz Y, invalid rows NaN = masked by the loss) + val rows +
legacy_val rows (dev_targets.py), the order prepare_alllab_view.py appends them.   usage: make_heads_view.py [dst]"""
import os, sys
from pathlib import Path
import numpy as np
ROOT = Path(os.environ.get("FUSION_ROOT") or Path(__file__).resolve().parents[3]); HERE = Path(__file__).resolve().parent
src = ROOT / "data/pod_history128_alllab_vf_v1"; dst = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data/pod_history128_alllab_vf_heads_v1"
dst.mkdir(parents=True, exist_ok=True)
skip = {"train_scalars.npy", "val_scalars.npy", "legacy_val_scalars.npy", "normalization.npz"}
for f in sorted(os.listdir(src)):
    if f not in skip and not (dst / f).exists():
        os.symlink((src / f).resolve(), dst / f)
t = np.load(ROOT / "results/analysis/scalar_probe_160/train.npz"); Ytr = t["Y"].astype(np.float64); Ytr[~t["valid"]] = np.nan
Yv, Yl = np.load(HERE / "dev_targets_val.npy"), np.load(HERE / "dev_targets_legacy_val.npy")
Y = np.concatenate([Ytr, Yv, Yl]); Y[~np.isfinite(Y)] = np.nan
m7, s7 = np.nanmean(Y, 0), np.maximum(np.nanstd(Y, 0), 1e-6)
with np.load(src / "normalization.npz", allow_pickle=True) as z:
    norm = {k: z[k] for k in z.files}
norm["scalar_mean"] = np.concatenate([norm["scalar_mean"], m7]).astype(np.float32); norm["scalar_std"] = np.concatenate([norm["scalar_std"], s7]).astype(np.float32)
np.savez(dst / "normalization.npz", **norm)
tr = np.load(src / "train_scalars.npy"); assert len(tr) == len(Y), (len(tr), len(Y))
np.save(dst / "train_scalars.npy", np.column_stack([tr, (Y - m7) / s7]).astype(np.float32))
for sp, Ys in (("val", Yv), ("legacy_val", Yl)):
    a = np.load(src / f"{sp}_scalars.npy"); assert len(a) == len(Ys), (sp, len(a), len(Ys))
    np.save(dst / f"{sp}_scalars.npy", np.column_stack([a, (Ys - m7) / s7]).astype(np.float32))
print(f"{dst}: train {len(Y)} rows x 9 scalars; scalar_mean {np.round(norm['scalar_mean'], 4).tolist()}")
