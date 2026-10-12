#!/usr/bin/env python3
"""v4 = v3 + KMODE=loc: radial compression about the axis LOCALISED in Z around the outboard (Rmax) and inboard (Rmin) LCFS vertices
(Gaussian in Z, sigma env KSIGZ m, default 0.25; each side only on its side of the axis), so kappa moves through the width extremes with a
small volume change and the X-point regions untouched.
Closed-loop elongation adjustment, v3 = v2 + deformation MODE (env KMODE): ap = area-preserving stretch (v1/v2), z = vertical stretch
only, r = radial compression only, zx = vertical stretch on the X-point side only (side = the LCFS vertex with the smaller |grad psi|),
smoothly switched on away from the axis height. v2 (jkappa_211): as lihead_210/kappa_adjust.py, plus a check of the REALISED official kappa of the
float32 field that is written (brentq can converge onto a jump of the separatrix extraction near the X-point, which overshoots): a root
whose float32 field misses the target by > KTOL is searched again on a 24-point scan of the bracket (sub-bracket roots, closest
admissible one), and the frame is reverted ('jump') when no admissible root exists or when tri_top/tri_bot/volume jump (> 0.02 abs / 2%). Area-preserving stretch about the magnetic axis, localised to the plasma:
    k_eff(R,Z) = 1 + w (k - 1),  w = 1 inside the LCFS (psi_N <= 1), smoothstep -> 0 at psi_N = TAPER (outside), 0 beyond;
    psi'(R, Z) = psi(R_ax + (R - R_ax) sqrt(k_eff), Z_ax + (Z - Z_ax) / sqrt(k_eff))     (bicubic resampling)
so the plasma is k times more elongated at fixed cross-section area and the far field is untouched. k is solved per frame (brentq) so
that the OFFICIAL kappa equals kappa_field + b (kappa_target - kappa_field). Frames whose extraction fails or whose axis moves by
> 1 mm are reverted. Unlike the li remap this moves the separatrix, so D_LCFS is part of the test.
usage: kappa_adjust.py <in.npz> <out.npz> <kappa_target.npy (frames, sorted-key order; NaN = skip)> <b> [workers] [taper=1.4]"""
import json, os, sys, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.optimize import brentq

ROOT = Path(__import__("os").environ.get("FUSION_ROOT") or Path(__file__).resolve().parents[3])
sys.path.insert(0, str(ROOT / "starter")); sys.path.insert(0, str(ROOT / "starter/fusion_scoring"))
import local_score as ls  # noqa: E402
from derive import derive_frame, _bilinear  # noqa: E402
from lcfs import extract_lcfs_with_sign  # noqa: E402
from o_point import find_o_point  # noqa: E402

mk = np.load(ROOT / "starter/fusion_scoring/masks/d3d_envelope.npz"); R, Z = mk["grid_R"].astype(float), mk["grid_Z"].astype(float)
mc = mk["mask_coarse"].astype(bool); mf = mc.astype(float); AX = ls.AXIS_SIGN["DIII-D"]
RR, ZZ = np.meshgrid(R, Z); dR, dZ = R[1] - R[0], Z[1] - Z[0]
PAY = TGT = None; B = 0.0; TAPER = 1.4; MODE = os.environ.get('KMODE', 'ap')


def init(pay, tgt, b, taper):
    global PAY, TGT, B, TAPER
    PAY, TGT, B, TAPER = np.load(pay), np.load(tgt), float(b), float(taper)


def shape_of(psi):
    try:
        d = derive_frame(psi, R, Z, "DIII-D", mc, mf, with_li=False, axis_sign=AX)
        return np.array([d.get(n, np.nan) for n in ("kappa", "tri_top", "tri_bot", "volume")], float)
    except Exception:
        return np.full(4, np.nan)


def kappa_of(psi):
    return float(shape_of(psi)[0])


KTOL = 2e-4


def adjust(psi, k_tgt):
    if not np.isfinite(k_tgt):
        return psi, "skip", 1.0
    C, _ = extract_lcfs_with_sign(psi, R, Z, "DIII-D", mc, mf, n_iter=ls.N_ITER, axis_sign=AX)
    if C is None:
        return psi, "skip", 1.0
    phi = AX * psi; ra, za, iz, ir, _ = find_o_point(phi, R, Z, mc); pax = float(phi[iz, ir])
    pb = float(np.median(_bilinear(phi, R, Z, np.asarray(C)[:, 0], np.asarray(C)[:, 1])))
    if not pax > pb:
        return psi, "skip", 1.0
    sN = (pax - phi) / (pax - pb); x = np.clip((sN - 1.0) / (TAPER - 1.0), 0.0, 1.0); w = np.where(sN <= 1.0, 1.0, 1.0 - x * x * (3 - 2 * x))

    if MODE == "zx":
        Cc = np.asarray(C); it, ib = int(np.argmax(Cc[:, 1])), int(np.argmin(Cc[:, 1]))
        gz, gr = np.gradient(phi, Z, R); gm = np.hypot(gz, gr)
        g_t, g_b = _bilinear(gm, R, Z, Cc[[it], 0], Cc[[it], 1])[0], _bilinear(gm, R, Z, Cc[[ib], 0], Cc[[ib], 1])[0]
        zx = Cc[ib, 1] if g_b <= g_t else Cc[it, 1]; u = np.clip((ZZ - za) / (zx - za), 0.0, 1.0) / 0.3
        side = np.clip(u, 0.0, 1.0); side = side * side * (3 - 2 * side)

    if MODE == "loc":
        Cc = np.asarray(C); io, ii = int(np.argmax(Cc[:, 0])), int(np.argmin(Cc[:, 0])); sz = float(os.environ.get("KSIGZ", "0.25"))
        lg = np.exp(-0.5 * ((ZZ - Cc[io, 1]) / sz) ** 2) * (RR >= ra) + np.exp(-0.5 * ((ZZ - Cc[ii, 1]) / sz) ** 2) * (RR < ra)

    if MODE in ("ri", "ro"):
        Cc = np.asarray(C); rmin, rmax = float(Cc[:, 0].min()), float(Cc[:, 0].max())
        u = np.clip((ra - RR) / (0.3 * (ra - rmin)), 0.0, 1.0) if MODE == "ri" else np.clip((RR - ra) / (0.3 * (rmax - ra)), 0.0, 1.0)
        rside = u * u * (3 - 2 * u)

    def field(k):
        ke = 1.0 + w * (k - 1.0)
        if MODE == "z":
            rs, zs = RR, za + (ZZ - za) / ke
        elif MODE == "r":
            rs, zs = ra + (RR - ra) * ke, ZZ
        elif MODE == "loc":
            kl = 1.0 + lg * (ke - 1.0); rs, zs = ra + (RR - ra) * kl, ZZ
        elif MODE in ("ri", "ro"):
            kr = 1.0 + rside * (ke - 1.0); rs, zs = ra + (RR - ra) * kr, ZZ
        elif MODE == "zx":
            kz = 1.0 + side * (ke - 1.0); rs, zs = RR, za + (ZZ - za) / kz
        else:
            rs, zs = ra + (RR - ra) * np.sqrt(ke), za + (ZZ - za) / np.sqrt(ke)
        return map_coordinates(psi, [(zs - Z[0]) / dZ, (rs - R[0]) / dR], order=3, mode="nearest")

    k0 = kappa_of(psi); kt = k0 + B * (k_tgt - k0)
    if not np.isfinite(k0) or abs(kt - k0) < 1e-5:
        return psi, "nochange", 1.0
    f = lambda k: kappa_of(field(k)) - kt  # noqa: E731
    try:
        span = 0.08 if MODE == 'ap' else (0.3 if MODE in ('ri', 'ro', 'zx', 'loc') else 0.16); lo, hi = (1.0, 1.0 + span) if kt > k0 else (1.0 - span, 1.0)
        flo, fhi = f(lo), f(hi)
        if not (np.isfinite(flo) and np.isfinite(fhi)) or flo * fhi > 0:
            return psi, "unreachable", 1.0
        k = brentq(f, lo, hi, xtol=1e-6, maxiter=40)
    except Exception:
        return psi, "fail", 1.0
    x0 = shape_of(psi)

    def ok_at(kk):
        nf = field(kk).astype(np.float32).astype(np.float64); xs = shape_of(nf)
        good = (np.isfinite(xs).all() and abs(xs[0] - kt) <= KTOL and abs(xs[1] - x0[1]) < 0.02 and abs(xs[2] - x0[2]) < 0.02
                and abs(xs[3] - x0[3]) < 0.02 * abs(x0[3]))
        return good, nf
    good, new = ok_at(k)
    if not good:
        grid = np.linspace(lo, hi, 25); fv = np.array([f(g) for g in grid]); cands = []
        for i in range(24):
            if np.isfinite(fv[i]) and np.isfinite(fv[i + 1]) and fv[i] * fv[i + 1] <= 0:
                try:
                    cands.append(brentq(f, grid[i], grid[i + 1], xtol=1e-7, maxiter=40))
                except Exception:
                    pass
        for kk in sorted(cands, key=lambda c: abs(c - 1.0)):
            good, new = ok_at(kk)
            if good:
                k = kk; break
        if not good:
            return psi, "jump", k
    a1 = find_o_point(AX * new, R, Z, mc)
    if np.hypot(a1[0] - ra, a1[1] - za) > 1e-3:
        return psi, "revert", k
    return new, "adjusted", k


def job(arg):
    key, first = arg; psi = PAY[key].astype(np.float64); out = np.empty(psi.shape, np.float32); st = []
    for j in range(len(psi)):
        try:
            new, s, k = adjust(psi[j], float(TGT[first + j]))
        except Exception:   # unexpected geometry: leave the frame unchanged
            new, s, k = psi[j], "error", 1.0
        out[j] = new; st.append((s, k))
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
    np.savez(dst, **out); names = [s for s, _ in st]; ks = [k for s, k in st if s == "adjusted"]
    summ = {"frames": len(st), "b": b, "taper": taper, "mode": MODE, **{s: names.count(s) for s in sorted(set(names))},
            "mean_abs_k_minus_1": float(np.mean(np.abs(np.array(ks) - 1))) if ks else 0.0, "seconds": round(time.monotonic() - t0)}
    Path(str(dst) + ".json").write_text(json.dumps(summ)); print(json.dumps(summ), flush=True)
