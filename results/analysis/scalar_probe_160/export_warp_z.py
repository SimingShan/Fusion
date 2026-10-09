#!/usr/bin/env python3
"""Z-ONLY / per-component version (env BAR, BAZ; regressor file env REG, default pred_direct_vf_test.npy).
Original docstring: Candidate export: INTERIOR-ONLY axis displacement (weight w = clip((psi-psi_b)/(psi_ax-psi_b),0,1)^POW: full at the
axis, zero at/outside the separatrix) toward beta_a * (direct-regressor axis - derived axis); boundary untouched.
Applied to the shipped 085 DIII-D test payload; MAST payload copied unchanged.
usage: export_warp.py <candidate e.g. 20261007-086> <beta_a> <POW>
Frame order: both the 085 payload (shot_%04d) and pred_direct_final_test.npy follow prediction_entries order.
"""
import json, shutil, sys, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np
from scipy.ndimage import map_coordinates

ROOT = Path(__import__("os").environ.get("FUSION_ROOT") or Path(__file__).resolve().parents[3])
sys.path.insert(0, str(ROOT / "starter")); sys.path.insert(0, str(ROOT / "starter/fusion_scoring"))
HERE = ROOT / "results/analysis/scalar_probe_160"
cand, BA, POW = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
CAND = ROOT / "results/candidates" / cand; CAND.mkdir(parents=True, exist_ok=True)
import os
SRC = Path(os.environ.get("WARP_SRC", str(ROOT / "results/candidates/20261006-085/diii_d_public_test.npz")))   # closed DIII-D test payload to warp
mk = np.load(ROOT / "starter/fusion_scoring/masks/d3d_envelope.npz")
R, Z = mk["grid_R"].astype(np.float64), mk["grid_Z"].astype(np.float64); mc = mk["mask_coarse"].astype(bool); mf = mc.astype(np.float64)
dR, dZ = R[1] - R[0], Z[1] - Z[0]; ZZ, RR = np.meshgrid(Z, R, indexing="ij")
BAR, BAZ = float(os.environ.get("BAR", BA)), float(os.environ.get("BAZ", BA)); REG = os.environ.get("REG", "pred_direct_vf_test.npy")
reg = np.load(HERE / REG).astype(np.float64)


def warp(p, dr, dz, pax, pb):
    w = np.clip((p - pb) / (pax - pb), 0.0, 1.0) ** POW
    return map_coordinates(p, [(ZZ - dz * w - Z[0]) / dZ, (RR - dr * w - R[0]) / dR], order=3, mode="nearest")


def work(args):
    import local_score as ls
    from derive import magnetic_axis
    from lcfs import extract_lcfs_with_sign
    key, f0 = args
    P = np.load(SRC)[key].astype(np.float64); out = np.empty_like(P); nwarp = 0
    for j in range(len(P)):
        i = f0 + j; p = P[j]
        ra, za, _, _ = magnetic_axis(p, R, Z, "DIII-D", mc, ls.AXIS_SIGN["DIII-D"])
        c, _ = extract_lcfs_with_sign(p, R, Z, "DIII-D", mc, mf, n_iter=ls.N_ITER, axis_sign=ls.AXIS_SIGN["DIII-D"])
        pax = map_coordinates(p, [[(za - Z[0]) / dZ], [(ra - R[0]) / dR]], order=3)[0] if np.isfinite([ra, za]).all() else np.nan
        pb = float(np.mean(map_coordinates(p, [(np.asarray(c)[:, 1] - Z[0]) / dZ, (np.asarray(c)[:, 0] - R[0]) / dR], order=1))) if c is not None else np.nan
        if np.isfinite([ra, za, pax, pb]).all() and pax != pb and np.isfinite(reg[i, :2]).all():
            dr, dz = BAR * (reg[i, 0] - ra), BAZ * (reg[i, 1] - za)
            out[j] = warp(p, dr, dz, pax, pb); nwarp += 1
        else:
            out[j] = p
    return key, out.astype(np.float32), nwarp


t0 = time.monotonic()
src = np.load(SRC); keys = sorted({k[:9] for k in src.files})
offs, f0 = {}, 0
for k in keys:
    offs[k] = f0; f0 += len(src[f"{k}_psirz"])
assert f0 == len(reg), (f0, len(reg))
with Pool(8) as pool:
    res = pool.map(work, [(f"{k}_psirz", offs[k]) for k in keys])
out, total = {}, 0
for key, arr, nw in res:
    out[key] = arr; total += nw
for k in keys:
    out[f"{k}_q95"] = src[f"{k}_q95"]; out[f"{k}_betaN"] = src[f"{k}_betaN"]
np.savez(CAND / "diii_d_public_test.npz", **out)
if (ROOT / "results/candidates/20261006-085/mast_public_test.npz").exists():   # dual entries only; the DIII-D package has no MAST payload
    shutil.copyfile(ROOT / "results/candidates/20261006-085/mast_public_test.npz", CAND / "mast_public_test.npz")
(CAND / "export.json").write_text(json.dumps({"base": str(SRC), "transform": "interior-only axis displacement toward direct-regressor axis (separatrix fixed)",
    "beta_r": BAR, "beta_z": BAZ, "pow": POW, "regressor": f"results/analysis/scalar_probe_160/{REG} (12 MLPs, train rows only)",
    "frames": f0, "warped_frames": total, "shots": len(keys), "seconds": time.monotonic() - t0}, indent=2))
print(f"wrote {CAND}/diii_d_public_test.npz frames={f0} warped={total} in {time.monotonic()-t0:.0f}s", flush=True)
