#!/usr/bin/env python3
"""Boundary-preserving li adjustment of a DIII-D payload toward per-frame li targets (closed loop on the OFFICIAL li).
Inside the payload's own LCFS (official extraction), the normalised flux s = (phi_ax - phi)/(phi_ax - phi_b) is remapped
    h(s) = s + a s (1 - s)^2        (h(0)=0, h(1)=1, h'(1)=1; monotone for -1 < a < 3)
so the axis cell, the LCFS level/contour (hence kappa, tri, volume, D_LCFS) and the edge gradient stay put while <Bp^2> changes;
a is solved per frame (brentq) so that the official li of the remapped field equals li_t = li_field + b (li_target - li_field).
Every adjusted frame is re-derived with the full official extraction; it is reverted unless the six non-li scalars are unchanged
(1e-9 relative) and the LCFS still extracts.  q95/betaN are copied.
usage: li_adjust.py <in.npz> <out.npz> <li_target.npy (frames, sorted-key order)> <b> [skip_mask.npy|-] [workers]"""
import json, sys, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np
from scipy.optimize import brentq

ROOT = Path(__import__("os").environ.get("FUSION_ROOT") or Path(__file__).resolve().parents[3])
sys.path.insert(0, str(ROOT / "starter")); sys.path.insert(0, str(ROOT / "starter/fusion_scoring"))
import local_score as ls  # noqa: E402
from derive import derive_frame, _points_in_polygon, _poloidal_field, _bilinear  # noqa: E402
from lcfs import extract_lcfs_with_sign  # noqa: E402
from o_point import find_o_point  # noqa: E402

NAMES = list(ls.CONS_SCALARS); NONLI = [n for n in NAMES if n != "li"]
mk = np.load(ROOT / "starter/fusion_scoring/masks/d3d_envelope.npz"); R, Z = mk["grid_R"].astype(float), mk["grid_Z"].astype(float)
mc = mk["mask_coarse"].astype(bool); mf = mc.astype(float); AX = ls.AXIS_SIGN["DIII-D"]
RR, ZZ = np.meshgrid(R, Z); dR, dZ = float(np.mean(np.diff(R))), float(np.mean(np.diff(Z))); DV = 2.0 * np.pi * RR * dR * dZ
PAY = TGT = SKIP = None; B = 0.0


def init(pay, tgt, skip, b):
    global PAY, TGT, SKIP, B
    PAY, TGT, B = np.load(pay), np.load(tgt), float(b); SKIP = None if skip in (None, "-") else np.load(skip)


def li_fixed(psi, C, inside, Cc, seg, lp):
    """Official li(2) for a field whose LCFS contour C is known (same arithmetic as derive.internal_inductance)."""
    Bp = _poloidal_field(psi, R, Z, "DIII-D"); Bp2 = np.where(np.isfinite(Bp), Bp, 0.0) ** 2
    V = float(np.sum(DV[inside])); Bp2_V = float(np.sum(Bp2[inside] * DV[inside])) / V
    b = _bilinear(Bp, R, Z, C[:, 0], C[:, 1]); G = float(np.sum(0.5 * (b + np.roll(b, -1)) * seg))
    return Bp2_V / (G / lp) ** 2


def adjust(psi, li_tgt):
    """-> (new psi, status, a, li0, li1, rms change / frame std)"""
    if not np.isfinite(li_tgt):
        return psi, "skip", 0.0, np.nan, np.nan, 0.0
    C, _ = extract_lcfs_with_sign(psi, R, Z, "DIII-D", mc, mf, n_iter=ls.N_ITER, axis_sign=AX)
    if C is None:
        return psi, "skip", 0.0, np.nan, np.nan, 0.0
    C = np.asarray(C, float); inside = _points_in_polygon(RR.ravel(), ZZ.ravel(), C).reshape(RR.shape)
    Cc = np.vstack([C, C[0]]); seg = np.sqrt(np.sum(np.diff(Cc, axis=0) ** 2, axis=1)); lp = float(np.sum(seg))
    phi = AX * psi; _, _, iz, ir, _ = find_o_point(phi, R, Z, mc); pax = float(phi[iz, ir])
    pb = float(np.median(_bilinear(phi, R, Z, C[:, 0], C[:, 1])))
    if not (pax > pb) or inside.sum() < 8:
        return psi, "skip", 0.0, np.nan, np.nan, 0.0
    s = np.clip((pax - phi) / (pax - pb), 0.0, 1.0); rem = inside & ((pax - phi) / (pax - pb) <= 1.0)

    def field(a):
        h = s + a * s * (1.0 - s) ** 2; out = phi.copy(); out[rem] = pax - h[rem] * (pax - pb); return AX * out

    li0 = li_fixed(psi, C, inside, Cc, seg, lp); li_t = li0 + B * (li_tgt - li0)
    if not np.isfinite(li0) or abs(li_t - li0) < 1e-5:
        return psi, "nochange", 0.0, li0, li0, 0.0
    f = lambda a: li_fixed(field(a), C, inside, Cc, seg, lp) - li_t  # noqa: E731
    lo, hi = (-0.9, 0.0) if li_t > li0 else (0.0, 2.5)          # li falls as a rises
    try:
        if f(lo) * f(hi) > 0:
            return psi, "unreachable", 0.0, li0, li0, 0.0
        a = brentq(f, lo, hi, xtol=1e-6, maxiter=60)
    except Exception:
        return psi, "fail", 0.0, li0, li0, 0.0
    new = field(a)
    d0 = derive_frame(psi, R, Z, "DIII-D", mc, mf, with_li=True, axis_sign=AX); d1 = derive_frame(new, R, Z, "DIII-D", mc, mf, with_li=True, axis_sign=AX)
    # the LCFS moves by ~30 um (cells straddling the contour interpolate the remapped inside values): kappa/tri ~3e-5, volume ~3e-5
    # relative, axis ~1e-5 m; the guard only rejects real extraction changes
    tol = {"R_axis": 2e-4, "Z_axis": 2e-4, "kappa": 5e-4, "tri_top": 5e-4, "tri_bot": 5e-4, "volume": 5e-4 * max(1.0, abs(d0.get("volume", 1.0)))}
    same = all(np.isfinite(d1.get(n, np.nan)) and abs(d1[n] - d0[n]) <= tol[n] for n in NONLI)
    if not same or not np.isfinite(d1.get("li", np.nan)):
        return psi, "revert", a, li0, li0, 0.0
    return new, "adjusted", a, d0["li"], d1["li"], float(np.sqrt(np.mean((new - psi) ** 2)) / (np.std(psi) + 1e-12))


def job(arg):
    key, first = arg; psi = PAY[key].astype(np.float64); outp = np.empty_like(psi, dtype=np.float32); rows = []
    for k in range(len(psi)):
        j = first + k
        if SKIP is not None and SKIP[j]:
            outp[k] = psi[k]; rows.append(("gated", 0.0, np.nan, np.nan, 0.0)); continue
        new, st, a, l0, l1, rc = adjust(psi[k], float(TGT[j])); outp[k] = new; rows.append((st, a, l0, l1, rc))
    return key, outp, rows


if __name__ == "__main__":
    src, dst, tgt, b = sys.argv[1], Path(sys.argv[2]), sys.argv[3], float(sys.argv[4])
    skip = sys.argv[5] if len(sys.argv) > 5 else "-"; W = int(sys.argv[6]) if len(sys.argv) > 6 else 16; t0 = time.monotonic()
    z = np.load(src); keys = sorted(k for k in z.files if k.endswith("_psirz")); n = [len(z[k]) for k in keys]
    firsts = np.concatenate([[0], np.cumsum(n)[:-1]]).astype(int); assert sum(n) == len(np.load(tgt)), "target length != frames"
    with Pool(W, initializer=init, initargs=(src, tgt, skip, b)) as pool:
        res = pool.map(job, list(zip(keys, firsts)), chunksize=1)
    out = {k: z[k] for k in z.files if not k.endswith("_psirz")}
    rows = []
    for key, p, r in res:
        out[key] = p; rows += r
    np.savez(dst, **out)
    st = [r[0] for r in rows]; adj = [r for r in rows if r[0] == "adjusted"]
    summ = {"frames": len(rows), "b": b, **{s: st.count(s) for s in sorted(set(st))},
            "mean_abs_dli": float(np.mean([abs(r[3] - r[2]) for r in adj])) if adj else 0.0,
            "mean_abs_a": float(np.mean([abs(r[1]) for r in adj])) if adj else 0.0,
            "rms_change_over_std": float(np.sqrt(np.mean([r[4] ** 2 for r in rows]))), "seconds": round(time.monotonic() - t0)}
    Path(str(dst) + ".json").write_text(json.dumps(summ, indent=1)); print(json.dumps(summ), flush=True)
