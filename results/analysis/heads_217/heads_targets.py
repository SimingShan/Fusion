#!/usr/bin/env python3
"""Targets from the HLVF members' consistency-scalar heads (predict time): H = mean over members of the heads sidecars
(<member>/test/diii_d_public_test_heads.npz, frames in payload order) -> <prefix>_heads.npy (N, 7: R_axis, Z_axis, kappa, tri_top, tri_bot,
volume, li); Z-shift regressor file <prefix>_zreg.npy = the gated direct predictions with the Z_axis column replaced by (H_Z + D_Z) / 2
(Thomson-poor frames stay NaN = no shift).   usage: heads_targets.py <members.txt (one test dir per line)> <direct_gated.npy> <out_prefix>"""
import json, sys
from pathlib import Path
import numpy as np
members = [l.split()[-1] for l in Path(sys.argv[1]).read_text().splitlines() if l.strip()]; Dg = np.load(sys.argv[2]).astype(np.float64); pre = sys.argv[3]
Hs = []
for m in members:
    z = np.load(Path(m) / "diii_d_public_test_heads.npz"); keys = sorted(z.files)
    Hs.append(np.concatenate([z[k] for k in keys]).astype(np.float64))
H = np.mean(Hs, 0); assert H.shape == Dg.shape, (H.shape, Dg.shape)
Z = Dg.copy(); Z[:, 1] = 0.5 * (H[:, 1] + Dg[:, 1])
np.save(pre + "_heads.npy", H); np.save(pre + "_zreg.npy", Z)
ok = np.isfinite(Dg[:, 1])
print(json.dumps({"members": len(members), "frames": int(len(H)), "z_targets": int(ok.sum()), "rms_HZ_minus_DZ": float(np.sqrt(np.nanmean((H[ok, 1] - Dg[ok, 1]) ** 2))),
                  "rms_Hkappa_minus_Dkappa": float(np.sqrt(np.nanmean((H[ok, 2] - Dg[ok, 2]) ** 2)))}))
