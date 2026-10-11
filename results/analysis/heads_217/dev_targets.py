#!/usr/bin/env python3
"""Official consistency scalars (R_axis, Z_axis, kappa, tri_top, tri_bot, volume, li; local_score derivation) of the LABEL maps of the
val and legacy_val rows that the all-labelled views append after the training rows (manifest order of data/advanced_history_v2).
Training rows use results/analysis/scalar_probe_160/train.npz (prepare.sh step 7).   -> dev_targets_{val,legacy_val}.npy (rows, 7)
usage: dev_targets.py [workers]"""
import json, os, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np
ROOT = Path(os.environ.get("FUSION_ROOT") or Path(__file__).resolve().parents[3])
sys.path.insert(0, str(ROOT / "starter")); sys.path.insert(0, str(ROOT / "starter/fusion_scoring"))
import local_score as ls  # noqa: E402
from derive import derive_frame  # noqa: E402
from lcfs import extract_lcfs_with_sign  # noqa: E402
mk = np.load(ROOT / "starter/fusion_scoring/masks/d3d_envelope.npz"); R, Z = mk["grid_R"].astype(float), mk["grid_Z"].astype(float)
mc = mk["mask_coarse"].astype(bool); mf = mc.astype(float); AX = ls.AXIS_SIGN["DIII-D"]
NAMES = ["R_axis", "Z_axis", "kappa", "tri_top", "tri_bot", "volume", "li"]


def one(path):
    z = np.load(path); out = []
    for p in z["psi"].astype(np.float64):
        try:
            ex, _ = extract_lcfs_with_sign(p, R, Z, "DIII-D", mc, mf, n_iter=ls.N_ITER, axis_sign=AX)
            v = derive_frame(p, R, Z, "DIII-D", mc, mf, contour=ex, with_li=True, axis_sign=AX); out.append([v.get(n, np.nan) for n in NAMES])
        except Exception:
            out.append([np.nan] * 7)
    return np.array(out, float)


if __name__ == "__main__":
    W = int(sys.argv[1]) if len(sys.argv) > 1 else 8; d = ROOT / "data/advanced_history_v2"; man = json.loads((d / "manifest.json").read_text())
    for split in ("val", "legacy_val"):
        entries = man.get("splits", man)[split]
        with Pool(W) as pool:
            Y = np.concatenate(pool.map(one, [str(d / e["path"]) for e in entries]))
        np.save(Path(__file__).parent / f"dev_targets_{split}.npy", Y); print(split, Y.shape, "rows without a boundary:", int((~np.isfinite(Y).all(1)).sum()), flush=True)
