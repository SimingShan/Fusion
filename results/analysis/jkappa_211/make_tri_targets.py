#!/usr/bin/env python3
"""Triangularity targets for tri_shift.py: J = weighted member mean of the members' own official tri_top/tri_bot (member_scalars.py output,
weights as stored), used only where J and the direct regressors D move the MEAN triangularity the same way relative to the base field
(Thomson-poor frames: D is NaN -> never). usage: make_tri_targets.py <direct.npy (N, 7)> <member_scalars.npz> <out.npy (N, 2)>"""
import sys
import numpy as np
D = np.load(sys.argv[1]).astype(np.float64); z = np.load(sys.argv[2]); S = z["S"].astype(np.float64); w = z["w"].astype(np.float64); base = z["base"].astype(np.float64)
fin = np.isfinite(S[..., 3]) & np.isfinite(S[..., 4]); Wk = np.where(fin, w[:, None], 0.0); den = Wk.sum(0)
J = np.stack([np.where(den >= 0.5 * w.sum(), (np.where(fin, S[..., c], 0.0) * Wk).sum(0) / np.maximum(den, 1e-12), np.nan) for c in (3, 4)], 1)
xm, Jm, Dm = base[:, 3:5].mean(1), J.mean(1), D[:, 3:5].mean(1)
ok = np.isfinite(xm) & np.isfinite(Jm) & np.isfinite(Dm) & (np.sign(Jm - xm) == np.sign(Dm - xm))
np.save(sys.argv[3], np.where(ok[:, None], J, np.nan)); print(f"tri targets {int(ok.sum())} of {len(ok)} frames; rms(Jm - xm) {np.sqrt(np.nanmean((Jm - xm)[ok] ** 2)):.5f}")
