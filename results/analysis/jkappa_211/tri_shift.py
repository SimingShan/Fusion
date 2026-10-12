#!/usr/bin/env python3
"""Closed-loop TRIANGULARITY shift through Rgeo (tri_top/tri_bot = (Rgeo - R_at_Zmax|Zmin) / a; R_at_Zmax/Zmin are grid-column quantised, Rgeo
and a are not). Both LCFS width extremes are moved by the SAME radial displacement delta (outboard Rmax and inboard Rmin both by +delta), so
the width a and kappa stay put and both triangularities change by ~ delta / a, through a deformation localised in Z around each extreme
(Gaussian sigma env TSIGZ m, default 0.25) that vanishes at the magnetic axis (linear ramps axis -> extreme) and outside the plasma:
    D_R = delta * w * g(Z) * h(R),  h = (R - R_ax)/(Rmax - R_ax) (R >= R_ax),  (R_ax - R)/(R_ax - Rmin) (R < R_ax);  psi'(R, Z) = psi(R - D_R, Z).
delta is solved per frame (brentq) so that the OFFICIAL mean triangularity equals mean_x + b (mean_target - mean_x); reverted unless kappa,
R_at_Zmax, R_at_Zmin are unchanged (1e-4 / 1 mm), volume within 1%, the axis within 1 mm. usage: tri_shift.py <in.npz> <out.npz> <tgt.npy (frames, 2:
tri_top, tri_bot targets; NaN = skip)> <b> [workers] [taper=1.4]"""
import json, os, sys, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.optimize import brentq
ROOT = Path(os.environ.get("FUSION_ROOT") or Path(__file__).resolve().parents[3])
sys.path.insert(0, str(ROOT / "starter")); sys.path.insert(0, str(ROOT / "starter/fusion_scoring"))
import local_score as ls  # noqa: E402
from derive import _bilinear, _shape_from_contour  # noqa: E402
from lcfs import extract_lcfs_with_sign  # noqa: E402
from o_point import find_o_point  # noqa: E402
mk = np.load(ROOT / "starter/fusion_scoring/masks/d3d_envelope.npz"); R, Z = mk["grid_R"].astype(float), mk["grid_Z"].astype(float)
mc = mk["mask_coarse"].astype(bool); mf = mc.astype(float); AX = ls.AXIS_SIGN["DIII-D"]; RR, ZZ = np.meshgrid(R, Z); dR, dZ = R[1] - R[0], Z[1] - Z[0]
PAY = TGT = None; B = 0.0; TAPER = 1.4; SZ = float(os.environ.get("TSIGZ", "0.25"))


def init(pay, tgt, b, taper):
    global PAY, TGT, B, TAPER
    PAY, TGT, B, TAPER = np.load(pay), np.load(tgt), float(b), float(taper)


def geo(psi):
    C, _ = extract_lcfs_with_sign(psi, R, Z, "DIII-D", mc, mf, n_iter=ls.N_ITER, axis_sign=AX)
    if C is None:
        return None, None
    C = np.asarray(C); s = _shape_from_contour(C)
    return C, np.array([s["kappa"], s["tri_top"], s["tri_bot"], s["volume"], C[int(np.argmax(C[:, 1])), 0], C[int(np.argmin(C[:, 1])), 0]], float)


def adjust(psi, tt, tb):
    if not (np.isfinite(tt) and np.isfinite(tb)):
        return psi, "skip"
    C, g0 = geo(psi)
    if C is None or not np.isfinite(g0).all():
        return psi, "skip"
    phi = AX * psi; ra, za, iz, ir, _ = find_o_point(phi, R, Z, mc); pax = float(phi[iz, ir])
    pb = float(np.median(_bilinear(phi, R, Z, C[:, 0], C[:, 1])))
    if not pax > pb:
        return psi, "skip"
    sN = (pax - phi) / (pax - pb); x = np.clip((sN - 1.0) / (TAPER - 1.0), 0.0, 1.0); w = np.where(sN <= 1.0, 1.0, 1.0 - x * x * (3 - 2 * x))
    io, ii = int(np.argmax(C[:, 0])), int(np.argmin(C[:, 0])); rmax, rmin = C[io, 0], C[ii, 0]
    if not (rmax > ra > rmin):
        return psi, "skip"
    out_side = RR >= ra
    h = np.where(out_side, (RR - ra) / (rmax - ra), (ra - RR) / (ra - rmin)); h = np.clip(h, 0.0, None)
    g = np.where(out_side, np.exp(-0.5 * ((ZZ - C[io, 1]) / SZ) ** 2), np.exp(-0.5 * ((ZZ - C[ii, 1]) / SZ) ** 2))
    prof = w * g * h

    def field(dl):
        return map_coordinates(psi, [(ZZ - Z[0]) / dZ, (RR - dl * prof - R[0]) / dR], order=3, mode="nearest")
    m0 = 0.5 * (g0[1] + g0[2]); mt = m0 + B * (0.5 * (tt + tb) - m0)
    if abs(mt - m0) < 1e-5:
        return psi, "nochange"
    def f(dl):
        _, gg = geo(field(dl)); return (0.5 * (gg[1] + gg[2]) - mt) if gg is not None else np.nan
    lo, hi = (0.0, 0.03) if mt > m0 else (-0.03, 0.0)
    try:
        flo, fhi = f(lo), f(hi)
        if not (np.isfinite(flo) and np.isfinite(fhi)) or flo * fhi > 0:
            return psi, "unreachable"
        dl = brentq(f, lo, hi, xtol=1e-6, maxiter=40)
    except Exception:
        return psi, "fail"
    new = field(dl).astype(np.float32).astype(np.float64); _, g1 = geo(new)
    KT = float(os.environ.get("TKTOL", "1e-4")); RELAX = os.environ.get("TRELAX", "0") == "1"
    if (g1 is None or not np.isfinite(g1).all() or abs(g1[0] - g0[0]) > KT or abs(g1[3] - g0[3]) > 0.01 * abs(g0[3])
            or (not RELAX and (abs(g1[4] - g0[4]) > 1e-3 or abs(g1[5] - g0[5]) > 1e-3))
            or (RELAX and abs(0.5 * (g1[1] + g1[2]) - mt) > 2e-3)):
        return psi, "jump"
    a1 = find_o_point(AX * new, R, Z, mc)
    if np.hypot(a1[0] - ra, a1[1] - za) > 1e-3:
        return psi, "revert"
    return new, "adjusted"


def job(arg):
    key, first = arg; psi = PAY[key].astype(np.float64); out = np.empty(psi.shape, np.float32); st = []
    for j in range(len(psi)):
        new, s = adjust(psi[j], float(TGT[first + j, 0]), float(TGT[first + j, 1])); out[j] = new; st.append(s)
    return key, out, st


if __name__ == "__main__":
    src, dst, tgt, b = sys.argv[1], Path(sys.argv[2]), sys.argv[3], float(sys.argv[4]); W = int(sys.argv[5]) if len(sys.argv) > 5 else 10
    taper = float(sys.argv[6]) if len(sys.argv) > 6 else 1.4; t0 = time.monotonic()
    z = np.load(src); keys = sorted(k for k in z.files if k.endswith("_psirz")); n = [len(z[k]) for k in keys]
    firsts = np.concatenate([[0], np.cumsum(n)[:-1]]).astype(int); assert sum(n) == len(np.load(tgt))
    with Pool(W, initializer=init, initargs=(src, tgt, b, taper)) as pool:
        res = pool.map(job, list(zip(keys, firsts)), chunksize=1)
    out = {k: z[k] for k in z.files if not k.endswith("_psirz")}; st = []
    for key, p, s in res:
        out[key] = p; st += s
    np.savez(dst, **out); summ = {"frames": len(st), "b": b, "taper": taper, "sigz": SZ, **{s: st.count(s) for s in sorted(set(st))}, "seconds": round(time.monotonic() - t0)}
    Path(str(dst) + ".json").write_text(json.dumps(summ)); print(json.dumps(summ), flush=True)
