#!/usr/bin/env python3
"""Error anatomy of the 31-member equal-weight POD-DeepONet blend (blend_gather_083 all_good).

Stages (all derived quantities via the OFFICIAL scorer functions in starter/fusion_scoring):
  frames    per-shot workers -> per-frame table for GT and blend + member disagreement
  ceilings  counterfactual psi replacements scored with official.score_shot/finalize_machine
  merge     gather per-shot outputs into frames.parquet / npz

Outputs under results/analysis/c1_error_anatomy_083/. Nothing existing is modified.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "starter"))
import evaluate  # noqa: E402  (scripts/evaluate.py: reference cache helpers)
import local_score as official  # noqa: E402
from common import AXIS_SIGN, CONS_SCALARS, N_CONS, N_SCALARS, SCALARS  # noqa: E402
from contour import symmetric_hausdorff  # noqa: E402
from derive import (_bilinear, _phi, _points_in_polygon, _poloidal_field,  # noqa: E402
                    derive_frame, internal_inductance, magnetic_axis)
from lcfs import extract_lcfs_with_sign  # noqa: E402

MACHINE = "DIII-D"
DATA_DIR = ROOT / "data/advanced_val_native_v3"
OUT_DIR = ROOT / "results/analysis/c1_error_anatomy_083"
MEMBERS_JSON = ROOT / "results/evaluations/blend_gather_083/members.json"
PSI_SIGN = 1.0                       # blend_gather_083 scored under psi_sign=+1
AXIS_SIGN_EFF = AXIS_SIGN[MACHINE] * PSI_SIGN
N_ITER, N_POINTS = 22, 512
IP_IDX = 20
COV_IDX = [76, 111, 130, 149]
N_PROF = 10
MU0 = 4e-7 * np.pi
LI_IDX = CONS_SCALARS.index("li")

# ----------------------------------------------------------------------------- helpers

def load_members():
    meta = json.loads(MEMBERS_JSON.read_text())
    names = meta["sets"]["all_good"]
    return names, [ROOT / meta["members"][n] for n in names]


def make_tasks():
    manifest = json.loads((DATA_DIR / "manifest.json").read_text())
    entries = manifest.get("splits", manifest)["val"]
    cache_dir = DATA_DIR / "scoring_cache"
    fp = evaluate.scorer_fingerprint()
    tasks = [{"path": str((DATA_DIR / e["path"]).resolve()), "validation_index": int(i),
              "max_frames": 0, "cache_dir": str(cache_dir.resolve()), "fingerprint": fp,
              "pred": None, "perfect": True, "n_frames": int(e["n_frames"]),
              "shot_index": e.get("shot_index"), "shot_id": e.get("shot_id")}
             for i, e in enumerate(entries)]
    return tasks


def load_gt_and_ref(task):
    gt, _, R, Z, coarse, fine, frames = evaluate.load_task(task)
    cache = evaluate.reference_path(task, gt, R, Z, frames)
    if not cache.exists():
        raise FileNotFoundError(f"reference cache missing: {cache}")
    ref = evaluate.read_reference(cache)
    with np.load(task["path"], allow_pickle=False) as shot:
        X = np.asarray(shot["X"], dtype=np.float32)
        times = np.asarray(shot["times"], dtype=np.float64)
    return gt, R, Z, coarse, fine, frames, ref, X, times


def load_member_stack(paths, vi, T):
    psis = np.empty((len(paths), T, 65, 65), dtype=np.float32)
    q95 = np.empty((len(paths), T)); bn = np.empty((len(paths), T))
    for m, p in enumerate(paths):
        with np.load(p, allow_pickle=False) as z:
            psis[m] = z[f"shot_{vi:04d}_psirz"]
            q95[m] = z[f"shot_{vi:04d}_q95"]
            bn[m] = z[f"shot_{vi:04d}_betaN"]
    return psis, q95, bn


# quadratic fit on a 3x3 patch: f = a + b x + c y + d x^2 + e xy + f y^2 (x = dR index, y = dZ index)
_XY = np.array([(x, y) for y in (-1, 0, 1) for x in (-1, 0, 1)], dtype=float)
_A = np.column_stack([np.ones(9), _XY[:, 0], _XY[:, 1], _XY[:, 0] ** 2, _XY[:, 0] * _XY[:, 1], _XY[:, 1] ** 2])
_PINV = np.linalg.pinv(_A)
_RING = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]


def find_saddles(phi, R, Z):
    """Discrete saddle cells (4 sign changes around the 8-ring) with quadratic sub-pixel refinement.

    Returns arrays (Rs, Zs, phis, iz, ir). Only cells whose fitted Hessian has det < 0 are kept.
    """
    c = phi[1:-1, 1:-1]
    n = phi.shape[0]
    signs = []
    for dz, dr in _RING:
        signs.append(np.sign(phi[1 + dz:n - 1 + dz, 1 + dr:n - 1 + dr] - c) >= 0)
    signs = np.stack(signs)                       # (8, n-2, n-2)
    changes = np.sum(signs != np.roll(signs, 1, axis=0), axis=0)
    cand = np.argwhere(changes >= 4)
    Rs, Zs, Ps, IZ, IR = [], [], [], [], []
    dR = float(R[1] - R[0]); dZ = float(Z[1] - Z[0])
    for (i, j) in cand:
        iz, ir = i + 1, j + 1
        patch = phi[iz - 1:iz + 2, ir - 1:ir + 2].ravel()
        a, b, cc, d, e, f = _PINV @ patch
        H = np.array([[2 * d, e], [e, 2 * f]])
        det = H[0, 0] * H[1, 1] - H[0, 1] * H[1, 0]
        if det >= 0:
            continue
        xy = -np.linalg.solve(H, np.array([b, cc]))
        xy = np.clip(xy, -1.0, 1.0)
        val = a + b * xy[0] + cc * xy[1] + d * xy[0] ** 2 + e * xy[0] * xy[1] + f * xy[1] ** 2
        Rs.append(R[ir] + xy[0] * dR); Zs.append(Z[iz] + xy[1] * dZ); Ps.append(val)
        IZ.append(iz); IR.append(ir)
    return (np.asarray(Rs), np.asarray(Zs), np.asarray(Ps), np.asarray(IZ, int), np.asarray(IR, int))


def contour_level(phi, R, Z, C):
    """LCFS level as the median bilinear sample of phi along the contour."""
    return float(np.median(_bilinear(phi, R, Z, C[:, 0], C[:, 1])))


def contour_min_mask(fine, R, Z, C):
    return float(np.min(_bilinear(fine, R, Z, C[:, 0], C[:, 1])))


def shape_stats(C):
    Rc, Zc = C[:, 0], C[:, 1]
    return dict(Rmin=float(Rc.min()), Rmax=float(Rc.max()), Zmin=float(Zc.min()), Zmax=float(Zc.max()),
                R_at_Zmax=float(Rc[int(np.argmax(Zc))]), R_at_Zmin=float(Rc[int(np.argmin(Zc))]))


def xpoint_analysis(phi, R, Z, C, level, amp, Za):
    """Pick the lower/upper saddle closest in flux level to the LCFS level; report their geometry."""
    Rs, Zs, Ps, _, _ = find_saddles(phi, R, Z)
    out = {}
    for side, sel in (("lo", Zs < Za if len(Zs) else np.zeros(0, bool)),
                      ("hi", Zs > Za if len(Zs) else np.zeros(0, bool))):
        if len(Zs) == 0 or not sel.any():
            out.update({f"x{side}_R": np.nan, f"x{side}_Z": np.nan, f"x{side}_dlev": np.nan,
                        f"x{side}_dist": np.nan, f"x{side}_phi": np.nan})
            continue
        dl = (Ps[sel] - level) / amp
        k = int(np.argmin(np.abs(dl)))
        Rk, Zk = Rs[sel][k], Zs[sel][k]
        dist = float(np.min(np.hypot(C[:, 0] - Rk, C[:, 1] - Zk)))
        out.update({f"x{side}_R": float(Rk), f"x{side}_Z": float(Zk), f"x{side}_dlev": float(dl[k]),
                    f"x{side}_dist": dist, f"x{side}_phi": float(Ps[sel][k])})
    return out


def classify_null(xd, min_mask, dR, tol_lev=0.01, tol_dist=2.0):
    lo = np.isfinite(xd["xlo_dlev"]) and abs(xd["xlo_dlev"]) < tol_lev and xd["xlo_dist"] < tol_dist * dR
    hi = np.isfinite(xd["xhi_dlev"]) and abs(xd["xhi_dlev"]) < tol_lev and xd["xhi_dist"] < tol_dist * dR
    if lo and hi:
        null = 3      # double null
    elif lo:
        null = 1      # lower single null
    elif hi:
        null = 2      # upper single null
    else:
        null = 0      # no saddle sets the level
    limited = 1 if (null == 0 and min_mask < 0.6) else (2 if null == 0 else 0)  # 1 = mask-limited, 2 = other
    return null, limited


def li_components(psi, R, Z, C):
    """Pieces of the official li(2): <Bp^2>_V, mu0 Ip / lp, V, lp (same formulas as derive.internal_inductance)."""
    if C is None or len(C) < 8:
        return np.nan, np.nan, np.nan, np.nan, None
    Rv = np.asarray(R, float); Zv = np.asarray(Z, float)
    Bp = _poloidal_field(psi, R, Z, MACHINE)
    RR, ZZ = np.meshgrid(Rv, Zv)
    inside = _points_in_polygon(RR.ravel(), ZZ.ravel(), C).reshape(RR.shape)
    if inside.sum() < 4:
        return np.nan, np.nan, np.nan, np.nan, inside
    dR = float(np.mean(np.diff(Rv))); dZ = float(np.mean(np.diff(Zv)))
    dV = 2.0 * np.pi * RR * dR * dZ
    Bp2 = np.where(np.isfinite(Bp), Bp, 0.0) ** 2
    V = float(np.sum(dV[inside]))
    Bp2_V = float(np.sum(Bp2[inside] * dV[inside])) / V
    Bp_on_C = _bilinear(Bp, Rv, Zv, C[:, 0], C[:, 1])
    Cc = np.vstack([C, C[0]])
    seg = np.sqrt(np.sum(np.diff(Cc, axis=0) ** 2, axis=1))
    G = float(np.sum(0.5 * (Bp_on_C + np.roll(Bp_on_C, -1)) * seg))
    lp = float(np.sum(seg))
    return Bp2_V, G / lp, V, lp, inside


def bp2_profile(psi, R, Z, inside, psi_axis, level_psi):
    """Mean Bp^2 in N_PROF bins of normalised flux s in [0,1] over grid points inside the LCFS."""
    Bp = _poloidal_field(psi, R, Z, MACHINE)
    Bp2 = np.where(np.isfinite(Bp), Bp, 0.0) ** 2
    den = level_psi - psi_axis
    s = (np.asarray(psi, float) - psi_axis) / den if den != 0 else np.full(psi.shape, np.nan)
    prof = np.full(N_PROF, np.nan)
    if inside is None:
        return prof
    sv = s[inside]; bv = Bp2[inside]
    b = np.clip((sv * N_PROF).astype(int), 0, N_PROF - 1)
    for k in range(N_PROF):
        m = b == k
        if m.any():
            prof[k] = bv[m].mean()
    return prof


def toroidal_current(psi, R, Z):
    """j_phi = -Delta* psi / (mu0 R), Delta* = d2/dR2 - (1/R) d/dR + d2/dZ2, 2nd-order central stencil."""
    p = np.asarray(psi, float)
    dR = float(R[1] - R[0]); dZ = float(Z[1] - Z[0])
    j = np.full(p.shape, np.nan)
    Rg = np.asarray(R, float)[None, 1:-1]
    d2R = (p[1:-1, 2:] - 2 * p[1:-1, 1:-1] + p[1:-1, :-2]) / dR ** 2
    d1R = (p[1:-1, 2:] - p[1:-1, :-2]) / (2 * dR)
    d2Z = (p[2:, 1:-1] - 2 * p[1:-1, 1:-1] + p[:-2, 1:-1]) / dZ ** 2
    j[1:-1, 1:-1] = -(d2R - d1R / Rg + d2Z) / (MU0 * Rg)
    return j


_KMAG = None


def hf_fraction(field, inside, kcut=10.0):
    """Fraction of 2-D spectral energy of (field * inside) beyond kcut cycles/box."""
    global _KMAG
    f = np.where(inside & np.isfinite(field), field, 0.0)
    if _KMAG is None:
        k = np.fft.fftfreq(65) * 65
        _KMAG = np.hypot(k[:, None], k[None, :])
    F = np.abs(np.fft.fft2(f)) ** 2
    tot = F.sum()
    return float(F[_KMAG > kcut].sum() / tot) if tot > 0 else np.nan


# ----------------------------------------------------------------------------- stage A worker

def frames_worker(task, member_paths, derive_members):
    gt, R, Z, coarse, fine, frames, ref, X, times = load_gt_and_ref(task)
    contours, cons_gt, cmask, rgeo = ref
    T = gt["psi"].shape[0]
    vi = task["validation_index"]
    psis, q95m, bnm = load_member_stack(member_paths, vi, T)
    M = len(member_paths)
    blend = np.mean(psis, axis=0, dtype=np.float64)            # (T,65,65)
    pstd = np.std(psis.astype(np.float64), axis=0)              # per-pixel member std
    blend_q95 = q95m.mean(0); blend_bn = bnm.mean(0)
    dR = float(R[1] - R[0])
    RR, ZZ = np.meshgrid(R, Z)

    cols = {}

    def put(k, name, v):
        cols.setdefault(name, np.full(T, np.nan))[k] = v

    gt_prof = np.full((T, N_PROF), np.nan); pr_prof = np.full((T, N_PROF), np.nan)
    pr_prof_gtc = np.full((T, N_PROF), np.nan)
    mem_cons = np.full((M, T, N_CONS), np.nan); mem_d = np.full((M, T), np.nan)
    err_sum = np.zeros((65, 65)); err_sq = np.zeros((65, 65)); abs_err_sum = np.zeros((65, 65))
    std_sum = np.zeros((65, 65))
    inside_gt_sum = np.zeros((65, 65))
    gpsi = gt["psi"].astype(np.float64)

    for k in range(T):
        pg, pp = gpsi[k], blend[k]
        C_gt = contours[k]
        e = pp - pg
        err_sum += e; err_sq += e * e; abs_err_sum += np.abs(e); std_sum += pstd[k]
        put(k, "sse_full", float(np.sum(e * e)))
        put(k, "gt_sum", float(pg.sum())); put(k, "gt_sumsq", float((pg * pg).sum()))
        put(k, "max_abs_err", float(np.abs(e).max()))
        iz_e, ir_e = np.unravel_index(int(np.argmax(np.abs(e))), e.shape)
        put(k, "max_err_R", float(R[ir_e])); put(k, "max_err_Z", float(Z[iz_e]))
        put(k, "psi_std_rms", float(np.sqrt(np.mean(pstd[k] ** 2))))
        put(k, "psi_std_max", float(pstd[k].max()))
        # inputs
        put(k, "Ip_kA", float(X[k, IP_IDX])); put(k, "time_ms", float(times[k]))
        put(k, "frac", k / (T - 1) if T > 1 else 0.0)
        put(k, "coverage", float(np.mean(X[k, COV_IDX])))
        put(k, "q95_gt", gt["q95"][k]); put(k, "betaN_gt", gt["betaN"][k])
        put(k, "q95_pred", blend_q95[k]); put(k, "betaN_pred", blend_bn[k])
        put(k, "q95_std", q95m[:, k].std()); put(k, "betaN_std", bnm[:, k].std())

        # ---- GT side (official functions) ----
        for j, name in enumerate(CONS_SCALARS):
            put(k, f"{name}_gt", cons_gt[k, j])
        Ra, Za, iz, ir = magnetic_axis(pg, R, Z, MACHINE, coarse)
        put(k, "iz_gt", iz); put(k, "ir_gt", ir)
        phi_g = _phi(pg, MACHINE)
        put(k, "psi_axis_gt", float(pg[iz, ir]))
        put(k, "err_at_axis", float(e[iz, ir])); put(k, "std_at_axis", float(pstd[k][iz, ir]))
        inside_g = None
        if C_gt is not None:
            lev_g = contour_level(phi_g, R, Z, C_gt)
            amp_g = phi_g[iz, ir] - lev_g
            put(k, "psi_bdry_gt", -lev_g); put(k, "amp_gt", amp_g)
            for n_, v in shape_stats(C_gt).items():
                put(k, f"{n_}_gt", v)
            mm = contour_min_mask(fine, R, Z, C_gt); put(k, "minmask_gt", mm)
            xd = xpoint_analysis(phi_g, R, Z, C_gt, lev_g, amp_g, Za)
            for n_, v in xd.items():
                put(k, f"{n_}_gt", v)
            null, lim = classify_null(xd, mm, dR); put(k, "null_gt", null); put(k, "limited_gt", lim)
            Bp2V, GL, V, lp, inside_g = li_components(pg, R, Z, C_gt)
            put(k, "Bp2V_gt", Bp2V); put(k, "GoverLp_gt", GL); put(k, "Vgrid_gt", V); put(k, "lp_gt", lp)
            if inside_g is not None:
                inside_gt_sum += inside_g
                put(k, "sse_in", float(np.sum((e * e)[inside_g])))
                put(k, "std_in_rms", float(np.sqrt(np.mean(pstd[k][inside_g] ** 2))))
                put(k, "mean_err_in", float(e[inside_g].mean()))
                gt_prof[k] = bp2_profile(pg, R, Z, inside_g, pg[iz, ir], -lev_g)
                pr_prof_gtc[k] = bp2_profile(pp, R, Z, inside_g, pg[iz, ir], -lev_g)
            if np.isfinite(xd["xlo_R"]):
                put(k, "err_at_xlo", float(_bilinear(pp, R, Z, np.array([xd["xlo_R"]]), np.array([xd["xlo_Z"]]))[0]
                                           - _bilinear(pg, R, Z, np.array([xd["xlo_R"]]), np.array([xd["xlo_Z"]]))[0]))
                put(k, "std_at_xlo", float(_bilinear(pstd[k], R, Z, np.array([xd["xlo_R"]]), np.array([xd["xlo_Z"]]))[0]))
            if np.isfinite(xd["xhi_R"]):
                put(k, "err_at_xhi", float(_bilinear(pp, R, Z, np.array([xd["xhi_R"]]), np.array([xd["xhi_Z"]]))[0]
                                           - _bilinear(pg, R, Z, np.array([xd["xhi_R"]]), np.array([xd["xhi_Z"]]))[0]))
            # li of the PRED field on the GT contour: isolates field error from contour error
            put(k, "li_pred_gtcontour", internal_inductance(pp, R, Z, MACHINE, C_gt))

        # ---- blend side (exactly the scorer's calls) ----
        ex, _ = extract_lcfs_with_sign(pp, R, Z, MACHINE, coarse, fine, n_iter=N_ITER, axis_sign=AXIS_SIGN_EFF)
        if C_gt is not None:
            put(k, "D_LCFS", 1.0 if ex is None else min(1.0, symmetric_hausdorff(ex, np.asarray(C_gt, float)) / rgeo))
        vals = derive_frame(pp, R, Z, MACHINE, coarse, fine, contour=ex, with_li=True, axis_sign=AXIS_SIGN_EFF)
        for name in CONS_SCALARS:
            put(k, f"{name}_pred", vals[name])
        Rap, Zap, izp, irp = magnetic_axis(pp, R, Z, MACHINE, coarse, AXIS_SIGN_EFF)
        put(k, "iz_pred", izp); put(k, "ir_pred", irp)
        phi_p = _phi(pp, MACHINE, AXIS_SIGN_EFF)
        put(k, "psi_axis_pred", float(pp[izp, irp]))
        if ex is not None:
            lev_p = contour_level(phi_p, R, Z, ex)
            amp_p = phi_p[izp, irp] - lev_p
            put(k, "psi_bdry_pred", -lev_p); put(k, "amp_pred", amp_p)
            for n_, v in shape_stats(ex).items():
                put(k, f"{n_}_pred", v)
            mm = contour_min_mask(fine, R, Z, ex); put(k, "minmask_pred", mm)
            xd = xpoint_analysis(phi_p, R, Z, ex, lev_p, amp_p, Zap)
            for n_, v in xd.items():
                put(k, f"{n_}_pred", v)
            null, lim = classify_null(xd, mm, dR); put(k, "null_pred", null); put(k, "limited_pred", lim)
            Bp2V, GL, V, lp, inside_p = li_components(pp, R, Z, ex)
            put(k, "Bp2V_pred", Bp2V); put(k, "GoverLp_pred", GL); put(k, "Vgrid_pred", V); put(k, "lp_pred", lp)
            if inside_p is not None:
                pr_prof[k] = bp2_profile(pp, R, Z, inside_p, pp[izp, irp], -lev_p)
                # toroidal current density inside the PREDICTED LCFS
                jg = toroidal_current(pg, R, Z); jp = toroidal_current(pp, R, Z)
                ok = inside_p & np.isfinite(jg) & np.isfinite(jp)
                if ok.sum() > 4:
                    dA = dR * float(Z[1] - Z[0])
                    put(k, "j_rms_gt", float(np.sqrt(np.mean(jg[ok] ** 2))))
                    put(k, "j_rms_diff", float(np.sqrt(np.mean((jp[ok] - jg[ok]) ** 2))))
                    put(k, "Ip_from_j_gt_kA", float(np.sum(jg[ok]) * dA / 1e3))
                    put(k, "Ip_from_j_pred_kA", float(np.sum(jp[ok]) * dA / 1e3))
                    put(k, "j_hf_gt", hf_fraction(jg, ok)); put(k, "j_hf_pred", hf_fraction(jp, ok))
                    put(k, "j_hf_diff", hf_fraction(jp - jg, ok))
                    # gradient-based roughness: RMS of the discrete Laplacian of the residual
                    put(k, "res_lap_rms", float(np.sqrt(np.mean((toroidal_current(e, R, Z)[ok]) ** 2))))

        # ---- member-level derived scalars (disagreement of the derived quantities) ----
        if derive_members:
            for m in range(M):
                pm = psis[m, k].astype(np.float64)
                exm, _ = extract_lcfs_with_sign(pm, R, Z, MACHINE, coarse, fine, n_iter=N_ITER, axis_sign=AXIS_SIGN_EFF)
                if C_gt is not None:
                    mem_d[m, k] = 1.0 if exm is None else min(1.0, symmetric_hausdorff(exm, np.asarray(C_gt, float)) / rgeo)
                vm = derive_frame(pm, R, Z, MACHINE, coarse, fine, contour=exm, with_li=True, axis_sign=AXIS_SIGN_EFF)
                for j, name in enumerate(CONS_SCALARS):
                    mem_cons[m, k, j] = vm[name]

    cols["validation_index"] = np.full(T, vi); cols["frame"] = np.arange(T)
    cols["shot_index"] = np.full(T, task["shot_index"] if task["shot_index"] is not None else -1)
    cols["T"] = np.full(T, T)
    cols["rgeo_ref"] = np.full(T, rgeo)
    for j, name in enumerate(CONS_SCALARS):
        cols[f"{name}_valid"] = cmask[:, j].astype(float)
    cols["has_ref_contour"] = np.array([c is not None for c in contours], float)

    out_dir = OUT_DIR / "per_shot"; out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / f"shot_{vi:04d}.npz", **cols, gt_prof=gt_prof, pr_prof=pr_prof, pr_prof_gtc=pr_prof_gtc,
             mem_cons=mem_cons, mem_d=mem_d, err_sum=err_sum, err_sq=err_sq, abs_err_sum=abs_err_sum,
             std_sum=std_sum, inside_gt_sum=inside_gt_sum)
    bdir = OUT_DIR / "blend"; bdir.mkdir(parents=True, exist_ok=True)
    np.savez(bdir / f"shot_{vi:04d}.npz", psirz=blend, q95=blend_q95, betaN=blend_bn)
    return vi, T


# ----------------------------------------------------------------------------- stage B: ceilings

def affine_fit(pp, pg):
    x = pp.ravel(); y = pg.ravel()
    A = np.column_stack([x, np.ones_like(x)])
    a, b = np.linalg.lstsq(A, y, rcond=None)[0]
    return a * pp + b


def gauss_smooth(p, sigma):
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(p, sigma, mode="nearest")


def feather(ind, sigma=1.0):
    """Soft 0..1 weight from a boolean indicator (Gaussian-feathered so no seam is introduced)."""
    from scipy.ndimage import gaussian_filter
    return np.clip(gaussian_filter(ind.astype(float), sigma, mode="nearest"), 0.0, 1.0)


def mix(w, pg, pp):
    return w * pg + (1.0 - w) * pp


def variants_for_frame(pp, pg, C_gt, R, Z, coarse, mean_err, inside_g, axis_cell, xlo):
    from scipy.ndimage import binary_dilation, binary_erosion
    v = {"blend": pp}
    if inside_g is not None:
        w_in = feather(inside_g)
        v["gt_inside_lcfs"] = mix(w_in, pg, pp)
        v["gt_outside_lcfs"] = mix(1.0 - w_in, pg, pp)
        ring = binary_dilation(inside_g, iterations=2) & ~binary_erosion(inside_g, iterations=2)
        v["gt_lcfs_ring2"] = mix(feather(ring), pg, pp)
        core = binary_erosion(inside_g, iterations=3)
        v["gt_core_eroded3"] = mix(feather(core), pg, pp)
    else:
        for n_ in ("gt_inside_lcfs", "gt_outside_lcfs", "gt_lcfs_ring2", "gt_core_eroded3"):
            v[n_] = pp
    iz, ir = axis_cell
    patch = np.zeros_like(pp, bool); patch[max(0, iz - 2):iz + 3, max(0, ir - 2):ir + 3] = True
    v["gt_axis_5x5"] = mix(feather(patch), pg, pp)
    xp = np.zeros_like(pp, bool)
    if xlo is not None:
        izx, irx = xlo
        xp[max(0, izx - 3):izx + 4, max(0, irx - 3):irx + 4] = True
    v["gt_xlo_7x7"] = mix(feather(xp), pg, pp)
    v["blend_plus_meanbias"] = pp + mean_err
    v["affine_per_frame"] = affine_fit(pp, pg)
    v["blend_smooth_s0p5"] = gauss_smooth(pp, 0.5)
    v["blend_smooth_s0p8"] = gauss_smooth(pp, 0.8)
    v["gt_outside_mask"] = mix(feather(~coarse), pg, pp)
    return v


VARIANTS = ["blend", "gt_inside_lcfs", "gt_outside_lcfs", "gt_lcfs_ring2", "gt_core_eroded3", "gt_axis_5x5",
            "gt_xlo_7x7", "blend_plus_meanbias", "affine_per_frame", "blend_smooth_s0p5", "blend_smooth_s0p8",
            "gt_outside_mask"]


def ceilings_worker(task, mean_err_path):
    gt, R, Z, coarse, fine, frames, ref, X, times = load_gt_and_ref(task)
    contours, cons_gt, cmask, rgeo = ref
    vi = task["validation_index"]
    with np.load(OUT_DIR / "blend" / f"shot_{vi:04d}.npz") as z:
        blend = z["psirz"]; bq = z["q95"]; bb = z["betaN"]
    with np.load(mean_err_path) as z:
        mean_err = z["mean_err"]          # GT - blend, averaged over all frames
    per = np.load(OUT_DIR / "per_shot" / f"shot_{vi:04d}.npz")
    T = blend.shape[0]
    gpsi = gt["psi"].astype(np.float64)
    RR, ZZ = np.meshgrid(R, Z)
    stacks = {name: np.empty_like(blend) for name in VARIANTS}
    for k in range(T):
        C_gt = contours[k]
        inside_g = None
        if C_gt is not None:
            inside_g = _points_in_polygon(RR.ravel(), ZZ.ravel(), C_gt).reshape(RR.shape)
        axis_cell = (int(per["iz_gt"][k]), int(per["ir_gt"][k]))
        xlo = None
        if np.isfinite(per["xlo_R_gt"][k]) and per["null_gt"][k] in (1, 3):
            xlo = (int(np.argmin(np.abs(Z - per["xlo_Z_gt"][k]))), int(np.argmin(np.abs(R - per["xlo_R_gt"][k]))))
        vs = variants_for_frame(blend[k], gpsi[k], C_gt, R, Z, coarse, mean_err, inside_g, axis_cell, xlo)
        for name in VARIANTS:
            stacks[name][k] = vs[name]
    # sufficient statistics (as evaluate.reference_worker)
    sums = {}
    sums["psi_sum"], sums["psi_sumsq"], sums["psi_n"] = evaluate.moment(gt["psi"])
    for prefix, values in (("scal", [gt[name] for name in SCALARS]),
                           ("cons", [cons_gt[:, j] for j in range(N_CONS)])):
        columns = np.asarray([evaluate.moment(v) for v in values], dtype=np.float64)
        for column, suffix in enumerate(("sum", "sumsq", "n")):
            sums[f"{prefix}_{suffix}"] = columns[:, column]
    return vi, sums, stacks, bq, bb


def score_variant_shot(task, name, psi, bq, bb, means):
    gt, R, Z, coarse, fine, frames, ref, X, times = load_gt_and_ref(task)
    pred = {"psirz": psi, "q95": bq, "betaN": bb}
    return name, official.score_shot(gt, ref, pred, R, Z, coarse, fine, PSI_SIGN, means)


def ceilings_shot_worker(task, mean_err_path, means):
    """Build all variants for one shot and score each with the official per-shot worker."""
    vi, sums, stacks, bq, bb = ceilings_worker(task, mean_err_path)
    gt, R, Z, coarse, fine, frames, ref, X, times = load_gt_and_ref(task)
    parts = {}
    for name in VARIANTS:
        pred = {"psirz": stacks[name], "q95": bq, "betaN": bb}
        parts[name] = official.score_shot(gt, ref, pred, R, Z, coarse, fine, PSI_SIGN, means)
    return vi, parts


def stats_worker(task):
    gt, R, Z, coarse, fine, frames, ref, X, times = load_gt_and_ref(task)
    cons_gt = ref[1]
    sums = {}
    sums["psi_sum"], sums["psi_sumsq"], sums["psi_n"] = evaluate.moment(gt["psi"])
    for prefix, values in (("scal", [gt[name] for name in SCALARS]),
                           ("cons", [cons_gt[:, j] for j in range(N_CONS)])):
        columns = np.asarray([evaluate.moment(v) for v in values], dtype=np.float64)
        for column, suffix in enumerate(("sum", "sumsq", "n")):
            sums[f"{prefix}_{suffix}"] = columns[:, column]
    return sums


def run_ceilings(workers):
    tasks = make_tasks()
    mean_err_path = OUT_DIR / "maps.npz"
    with ProcessPoolExecutor(max_workers=workers) as ex:
        stats_list = list(ex.map(stats_worker, tasks))
    stats = {key: sum(s[key] for s in stats_list) for key in stats_list[0]}
    means = (stats["psi_sum"] / stats["psi_n"],
             np.divide(stats["scal_sum"], stats["scal_n"], out=np.zeros(N_SCALARS), where=stats["scal_n"] > 0),
             np.divide(stats["cons_sum"], stats["cons_n"], out=np.zeros(N_CONS), where=stats["cons_n"] > 0))
    order = sorted(range(len(tasks)), key=lambda i: -tasks[i]["n_frames"])
    accs = {name: official.Accum() for name in VARIANTS}
    for a in accs.values():
        a.psi_sign = PSI_SIGN
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(ceilings_shot_worker, tasks[i], mean_err_path, means): i for i in order}
        for n, f in enumerate(as_completed(futs), 1):
            vi, parts = f.result()
            for name in VARIANTS:
                accs[name].add(parts[name])
            print(f"ceilings: {n}/{len(tasks)} shots ({time.time() - t0:.0f}s)", flush=True)
    results = {name: official.finalize_machine(accs[name], stats) for name in VARIANTS}
    base = results["blend"]["S"]
    for name in VARIANTS:
        results[name]["delta_S_vs_blend"] = round(results[name]["S"] - base, 6)
    (OUT_DIR / "ceilings.json").write_text(json.dumps(evaluate.json_safe(results), indent=2) + "\n")
    for name in VARIANTS:
        r = results[name]
        print(f"{name:22s} S={r['S']:.6f} dS={r['delta_S_vs_blend']:+.6f} r2psi={r['r2_psi']:.6f} "
              f"qb={r['r2_qb']:.6f} dlcfs={r['dlcfs']:.6f} cons={r['consistency']:.6f} "
              + " ".join(f"{k}={v:.4f}" for k, v in r["r2_cons_each"].items()), flush=True)


# ----------------------------------------------------------------------------- stage A driver + merge

def run_frames(workers, derive_members):
    tasks = make_tasks()
    names, paths = load_members()
    print(f"{len(tasks)} shots, {len(paths)} members, derive_members={derive_members}", flush=True)
    order = sorted(range(len(tasks)), key=lambda i: -tasks[i]["n_frames"])
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(frames_worker, tasks[i], paths, derive_members): i for i in order}
        done_frames = 0
        for n, f in enumerate(as_completed(futs), 1):
            vi, T = f.result()
            done_frames += T
            print(f"frames: {n}/{len(tasks)} shots, {done_frames} frames ({time.time() - t0:.0f}s)", flush=True)
    merge()


def merge():
    import pandas as pd
    tasks = make_tasks()
    names, _ = load_members()
    rows = []; gt_prof = []; pr_prof = []; pr_prof_gtc = []; mem_cons = []; mem_d = []
    err_sum = np.zeros((65, 65)); err_sq = np.zeros((65, 65)); abs_err_sum = np.zeros((65, 65))
    std_sum = np.zeros((65, 65)); inside_sum = np.zeros((65, 65)); n = 0
    for t in tasks:
        z = np.load(OUT_DIR / "per_shot" / f"shot_{t['validation_index']:04d}.npz")
        T = int(z["T"][0])
        rows.append({k: z[k] for k in z.files if z[k].ndim == 1 and z[k].shape[0] == T})
        gt_prof.append(z["gt_prof"]); pr_prof.append(z["pr_prof"]); pr_prof_gtc.append(z["pr_prof_gtc"])
        mem_cons.append(z["mem_cons"]); mem_d.append(z["mem_d"])
        err_sum += z["err_sum"]; err_sq += z["err_sq"]; abs_err_sum += z["abs_err_sum"]; std_sum += z["std_sum"]
        inside_sum += z["inside_gt_sum"]; n += T
    keys = sorted(set().union(*[r.keys() for r in rows]))
    df = pd.DataFrame({k: np.concatenate([r.get(k, np.full(len(r["frame"]), np.nan)) for r in rows]) for k in keys})
    df.to_parquet(OUT_DIR / "frames.parquet", index=False)
    np.savez(OUT_DIR / "profiles.npz", gt_prof=np.concatenate(gt_prof), pr_prof=np.concatenate(pr_prof),
             pr_prof_gtc=np.concatenate(pr_prof_gtc))
    np.savez(OUT_DIR / "members_derived.npz", mem_cons=np.concatenate(mem_cons, axis=1),
             mem_d=np.concatenate(mem_d, axis=1), names=np.array(names))
    np.savez(OUT_DIR / "maps.npz", mean_err=-err_sum / n, rms_err=np.sqrt(err_sq / n), mean_abs_err=abs_err_sum / n,
             mean_std=std_sum / n, inside_frac=inside_sum / n, n_frames=n)
    print(f"merged {n} frames -> {OUT_DIR / 'frames.parquet'}; cols={len(keys)}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=["frames", "merge", "ceilings", "all"])
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--no-member-derive", action="store_true")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.stage in ("frames", "all"):
        run_frames(args.workers, not args.no_member_derive)
    if args.stage == "merge":
        merge()
    if args.stage in ("ceilings", "all"):
        run_ceilings(args.workers)


if __name__ == "__main__":
    main()
