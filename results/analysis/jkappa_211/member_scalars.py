#!/usr/bin/env python3
"""Official derived scalars (R_axis, Z_axis, kappa, tri_top, tri_bot, volume, li) of every member payload (and optionally of a base
payload), per frame in sorted-key order (the row order of the li_adjust/kappa_adjust targets). Prediction-side derivation exactly as the
scorer: contour = extract_lcfs_with_sign(n_iter=N_ITER, axis_sign=DIII-D), then derive_frame(contour=that, axis_sign=DIII-D).
usage: member_scalars.py <members.txt (lines 'weight path')> <out.npz> [base.npz|-] [workers]   [env NO_LI=1: members without li]
-> out.npz: S (K, N, 7) float32 (NaN = failed), w (K,), base (N, 7) if given, keys, counts"""
import os, sys, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np

ROOT = Path(__import__("os").environ.get("FUSION_ROOT") or Path(__file__).resolve().parents[3])
sys.path.insert(0, str(ROOT / "starter")); sys.path.insert(0, str(ROOT / "starter/fusion_scoring"))
import local_score as ls  # noqa: E402
from derive import derive_frame  # noqa: E402
from lcfs import extract_lcfs_with_sign  # noqa: E402

NAMES = ["R_axis", "Z_axis", "kappa", "tri_top", "tri_bot", "volume", "li"]
mk = np.load(ROOT / "starter/fusion_scoring/masks/d3d_envelope.npz"); R, Z = mk["grid_R"].astype(float), mk["grid_Z"].astype(float)
mc = mk["mask_coarse"].astype(bool); mf = mc.astype(np.float64); AX = ls.AXIS_SIGN["DIII-D"]; WITH_LI = os.environ.get("NO_LI", "0") != "1"


def scalars(psi, with_li=None):
    psi = psi.astype(np.float64); with_li = WITH_LI if with_li is None else with_li
    try:
        ex, _ = extract_lcfs_with_sign(psi, R, Z, "DIII-D", mc, mf, n_iter=ls.N_ITER, axis_sign=AX)
        v = derive_frame(psi, R, Z, "DIII-D", mc, mf, contour=ex, with_li=with_li, axis_sign=AX)
        return [v.get(n, np.nan) for n in NAMES]
    except Exception:
        return [np.nan] * 7


def job(arg):
    k, path, key, is_base = arg
    with np.load(path) as z:
        P = z[key]
    return k, key, np.array([scalars(P[j], True if is_base else None) for j in range(len(P))], np.float32)


if __name__ == "__main__":
    t0 = time.monotonic(); lines = [l.split() for l in Path(sys.argv[1]).read_text().splitlines() if l.strip()]
    w = np.array([float(a) for a, _ in lines]); paths = [p for _, p in lines]; out = Path(sys.argv[2])
    base = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3] != "-" else None; W = int(sys.argv[4]) if len(sys.argv) > 4 else 10
    with np.load(paths[0]) as z:
        keys = sorted(k for k in z.files if k.endswith("_psirz")); counts = [len(z[k]) for k in keys]
    first = dict(zip(keys, np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(int))); N = int(sum(counts))
    srcs = paths + ([base] if base else []); S = np.full((len(srcs), N, 7), np.nan, np.float32)
    jobs = [(k, p, key, bool(base) and k == len(paths)) for k, p in enumerate(srcs) for key in keys]   # the base always gets li (env NO_LI=1 skips it for members)
    with Pool(W) as pool:
        for i, (k, key, arr) in enumerate(pool.imap_unordered(job, jobs, chunksize=1)):
            S[k, first[key]:first[key] + len(arr)] = arr
            if (i + 1) % 200 == 0:
                print(f"{i + 1}/{len(jobs)} [{time.monotonic() - t0:.0f}s]", flush=True)
    res = {"S": S[:len(paths)], "w": w, "keys": np.array(keys), "counts": np.array(counts)}
    if base:
        res["base"] = S[-1]
    np.savez(out, **res); print(f"done {out}: {len(paths)} members x {N} frames, failed {np.isnan(S[..., 2]).mean():.4f} [{time.monotonic() - t0:.0f}s]", flush=True)
