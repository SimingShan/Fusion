#!/usr/bin/env python3
"""Targets for the post-processing chain from per-member official scalars (member_scalars.py outputs, possibly in parts):
  kappa target  M = (J + D) / 2,  J = weighted member mean of member-derived kappa, D = direct regressors' kappa (gate-excluded predictions:
                NaN rows = Thomson-poor frames, never adjusted);
  li target     li_t = li_base + LIB (li_D - li_base), anchored on the official li of the PRE-kappa field (the 'base' of the first part),
                realised with li_adjust.py at b = 1 after the kappa step (so the stretch's li side effect is removed).
usage: make_targets.py <direct.npy (N, 7)> <out_prefix> <mscal part.npz[@member_weight] ...>   [env LIB=0.35, AGREE=1: kappa targets only where sign(J-x) = sign(D-x)]
(member_weight overrides the file's per-member weights, e.g. to restore the blend's family shares from a member subset)
-> <out_prefix>_kappa.npy, <out_prefix>_li.npy, <out_prefix>_summary.json"""
import json, os, sys
from pathlib import Path
import numpy as np

D = np.load(sys.argv[1]).astype(np.float64); pre = sys.argv[2]; parts = sys.argv[3:]; LIB = float(os.environ.get("LIB", "0.35"))
S, w, base = [], [], None
for p in parts:
    p, _, mw = p.partition("@"); z = np.load(p); S.append(z["S"].astype(np.float64)); w.append(np.full(len(z["w"]), float(mw)) if mw else z["w"])
    if "base" in z.files:
        base = z["base"].astype(np.float64)
S, w = np.concatenate(S), np.concatenate(w); assert base is not None and S.shape[1] == len(D) == len(base), (S.shape, D.shape)
fin = np.isfinite(S[..., 2]); Wk = np.where(fin, w[:, None], 0.0); den = Wk.sum(0)
J = np.where(den >= 0.5 * w.sum(), (np.where(fin, S[..., 2], 0.0) * Wk).sum(0) / np.maximum(den, 1e-12), np.nan)
ok = np.isfinite(J) & np.isfinite(D[:, 2]) & np.isfinite(base[:, 2])
if os.environ.get("AGREE", "0") == "1":     # only where the member mean and the direct regressors move kappa the same way
    ok &= np.sign(J - base[:, 2]) == np.sign(D[:, 2] - base[:, 2])
M = np.where(ok, 0.5 * (J + D[:, 2]), np.nan)
li_ok = np.isfinite(D[:, 6]) & np.isfinite(base[:, 6]); li_t = np.where(li_ok, base[:, 6] + LIB * (D[:, 6] - base[:, 6]), np.nan)
np.save(pre + "_kappa.npy", M); np.save(pre + "_li.npy", li_t)
summ = {"frames": int(len(D)), "members": int(len(S)), "agree_gate": os.environ.get("AGREE", "0") == "1", "kappa_targets": int(ok.sum()), "li_targets": int(li_ok.sum()),
        "rms_M_minus_base": float(np.sqrt(np.nanmean((M - base[:, 2]) ** 2))), "rms_J_minus_base": float(np.sqrt(np.nanmean((J - base[:, 2]) ** 2))),
        "rms_D_minus_base": float(np.sqrt(np.nanmean((D[:, 2] - base[:, 2]) ** 2)))}
Path(pre + "_summary.json").write_text(json.dumps(summ, indent=1)); print(json.dumps(summ))
