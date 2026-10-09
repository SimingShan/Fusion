#!/usr/bin/env python3
"""Stage C of the error anatomy: decompose the blend's residual from the per-frame table.

Reads results/analysis/c1_error_anatomy_083/{frames.parquet, profiles.npz, members_derived.npz,
maps.npz, ceilings.json}; writes report.json and REPORT.md next to them. Login-node cheap.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy import stats as sst

ROOT = Path(__file__).resolve().parents[2]
import os
OUT = Path(os.environ.get("C1_ANAT_OUT", ROOT / "results/analysis/c1_error_anatomy_083"))
CONS = ["R_axis", "Z_axis", "kappa", "tri_top", "tri_bot", "volume", "li"]
W = dict(psi=0.55, qb=0.15, lcfs=0.10, cons=0.20)
NULL_NAMES = {0: "no-saddle", 1: "lower-SN", 2: "upper-SN", 3: "double-null"}


def r2_pooled(g, p, valid):
    m = valid & np.isfinite(g) & np.isfinite(p)
    ss_res = float(np.sum((g[m] - p[m]) ** 2))
    ss_tot = float(np.sum((g[m] - g[m].mean()) ** 2))
    return 1 - ss_res / ss_tot, ss_res, ss_tot, int(m.sum())


def group_shares(err2, weight, key, labels=None):
    """Share of the weighted SS_res per group vs share of frames."""
    df = pd.DataFrame({"e": err2 * weight, "k": key})
    df = df[np.isfinite(df.e)]
    g = df.groupby("k")["e"]
    tot = df.e.sum()
    out = []
    for k, v in g.sum().items():
        n = int((df.k == k).sum())
        out.append({"group": (labels[k] if labels and k in labels else str(k)), "n": n,
                    "frame_share": round(n / len(df), 4), "ss_share": round(float(v / tot), 4),
                    "mean_err2_ratio": round(float((v / n) / (tot / len(df))), 3)})
    return out


def qbins(x, q):
    edges = np.nanquantile(x, np.linspace(0, 1, q + 1))
    edges[-1] += 1e-9
    b = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, q - 1).astype(float)
    b[~np.isfinite(x)] = np.nan
    return b, edges


def ols(y, Xcols, names):
    m = np.isfinite(y)
    for c in Xcols:
        m &= np.isfinite(c)
    X = np.column_stack([np.ones(m.sum())] + [c[m] for c in Xcols])
    yy = y[m]
    beta, *_ = np.linalg.lstsq(X, yy, rcond=None)
    resid = yy - X @ beta
    dof = len(yy) - X.shape[1]
    s2 = resid @ resid / dof
    cov = s2 * np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    return {"n": int(m.sum()), "coef": {n: round(float(b), 6) for n, b in zip(["const"] + names, beta)},
            "t": {n: round(float(b / s), 2) for n, b, s in zip(["const"] + names, beta, se)},
            "ss_before": float(yy @ yy), "ss_after": float(resid @ resid),
            "frac_ss_removed": round(1 - float(resid @ resid) / float(yy @ yy), 4)}


def main():
    df = pd.read_parquet(OUT / "frames.parquet")
    prof = np.load(OUT / "profiles.npz")
    mem = np.load(OUT / "members_derived.npz")
    maps = np.load(OUT / "maps.npz")
    ceil = json.loads((OUT / "ceilings.json").read_text()) if (OUT / "ceilings.json").exists() else {}
    N = len(df)
    rep = {"n_frames": N, "n_shots": int(df.validation_index.nunique())}

    # ------------------------------------------------------------------ 0. reproduce the score
    base = {}
    for s in CONS:
        r2, ssr, sst_, n = r2_pooled(df[f"{s}_gt"].values, df[f"{s}_pred"].values, df[f"{s}_valid"].values > 0)
        base[s] = {"r2": round(r2, 6), "ss_res": ssr, "ss_tot": sst_, "n": n,
                   "rmse": round(float(np.sqrt(ssr / n)), 6), "gt_std": round(float(np.sqrt(sst_ / n)), 6)}
    for s in ("q95", "betaN"):
        r2, ssr, sst_, n = r2_pooled(df[f"{s}_gt"].values, df[f"{s}_pred"].values, np.ones(N, bool))
        base[s] = {"r2": round(r2, 6), "ss_res": ssr, "ss_tot": sst_, "n": n, "rmse": round(float(np.sqrt(ssr / n)), 6)}
    mean_psi = df.gt_sum.sum() / (N * 65 * 65)
    ss_tot_psi = df.gt_sumsq.sum() - N * 65 * 65 * mean_psi ** 2
    base["psi"] = {"r2": round(1 - df.sse_full.sum() / ss_tot_psi, 6), "ss_res": float(df.sse_full.sum()),
                   "ss_res_inside_lcfs_share": round(float(df.sse_in.sum() / df.sse_full.sum()), 4)}
    shot_d = df.groupby("validation_index").D_LCFS.mean()
    base["D_LCFS"] = {"mean_of_shot_means": round(float(shot_d.mean()), 6), "frame_mean": round(float(df.D_LCFS.mean()), 6),
                      "median": round(float(df.D_LCFS.median()), 6), "p90": round(float(df.D_LCFS.quantile(.9)), 6),
                      "p99": round(float(df.D_LCFS.quantile(.99)), 6), "max": round(float(df.D_LCFS.max()), 6)}
    cons = np.mean([max(0, base[s]["r2"]) for s in CONS])
    S = W["psi"] * base["psi"]["r2"] + W["qb"] * np.mean([base["q95"]["r2"], base["betaN"]["r2"]]) \
        + W["lcfs"] * (1 - base["D_LCFS"]["mean_of_shot_means"]) + W["cons"] * cons
    base["S_reconstructed"] = round(float(S), 6); base["consistency"] = round(float(cons), 6)
    # S loss per scalar: 0.20/7 * (1 - r2)
    base["S_loss_by_term"] = {**{s: round(W["cons"] / 7 * (1 - base[s]["r2"]), 6) for s in CONS},
                              "D_LCFS": round(W["lcfs"] * base["D_LCFS"]["mean_of_shot_means"], 6),
                              "q95": round(W["qb"] / 2 * (1 - base["q95"]["r2"]), 6),
                              "betaN": round(W["qb"] / 2 * (1 - base["betaN"]["r2"]), 6),
                              "psi": round(W["psi"] * (1 - base["psi"]["r2"]), 6)}
    rep["baseline"] = base

    # ------------------------------------------------------------------ 1. member disagreement of derived scalars
    mc = mem["mem_cons"]                     # (M, N, 7)
    M = mc.shape[0]
    mem_std = np.nanstd(mc, axis=0)          # (N,7)
    mem_mean = np.nanmean(mc, axis=0)
    for j, s in enumerate(CONS):
        df[f"{s}_memstd"] = mem_std[:, j]
        df[f"{s}_memmean"] = mem_mean[:, j]
    df["D_memmean"] = np.nanmean(mem["mem_d"], axis=0)
    df["D_memstd"] = np.nanstd(mem["mem_d"], axis=0)
    errs = {s: (df[f"{s}_pred"] - df[f"{s}_gt"]).values for s in CONS}
    errs["q95"] = (df.q95_pred - df.q95_gt).values; errs["betaN"] = (df.betaN_pred - df.betaN_gt).values
    valid = {s: (df[f"{s}_valid"].values > 0) & np.isfinite(errs[s]) for s in CONS}
    valid["q95"] = np.isfinite(errs["q95"]); valid["betaN"] = np.isfinite(errs["betaN"])
    dis = {}
    for s in CONS:
        v = valid[s] & np.isfinite(df[f"{s}_memstd"].values)
        e = np.abs(errs[s][v]); sd = df[f"{s}_memstd"].values[v]
        r_p = float(np.corrcoef(e, sd)[0, 1]); r_s = float(sst.spearmanr(e, sd).correlation)
        r2_e2 = float(np.corrcoef(e ** 2, sd ** 2)[0, 1])
        # share of SS_res in the top-decile disagreement frames
        top = sd >= np.quantile(sd, 0.9)
        # member-mean-of-scalars vs blend: does averaging psi beat averaging scalars?
        mm = df[f"{s}_memmean"].values[v]
        g = df[f"{s}_gt"].values[v]
        ss_blend = float(np.sum(e ** 2)); ss_memmean = float(np.sum((mm - g) ** 2))
        ss_single = float(np.nanmean([np.nansum((mc[m][v, CONS.index(s)] - g) ** 2) for m in range(M)]))
        # bias vs variance: blend error RMS vs member std / sqrt(M) (pure-variance expectation)
        dis[s] = {"pearson_abs_err_vs_memstd": round(r_p, 4), "spearman": round(r_s, 4),
                  "pearson_err2_vs_var": round(r2_e2, 4),
                  "ss_share_top10pct_disagreement": round(float(np.sum(e[top] ** 2) / ss_blend), 4),
                  "rms_blend_err": round(float(np.sqrt(np.mean(e ** 2))), 6),
                  "rms_member_std": round(float(np.sqrt(np.mean(sd ** 2))), 6),
                  "rms_member_std_over_sqrtM": round(float(np.sqrt(np.mean(sd ** 2)) / np.sqrt(M)), 6),
                  "ss_res_blend": ss_blend, "ss_res_mean_of_member_scalars": ss_memmean,
                  "ss_res_avg_single_member": ss_single,
                  "r2_blend": round(1 - ss_blend / base[s]["ss_tot"], 6),
                  "r2_mean_of_member_scalars": round(1 - ss_memmean / base[s]["ss_tot"], 6),
                  "r2_avg_single_member": round(1 - ss_single / base[s]["ss_tot"], 6)}
    v = np.isfinite(df.psi_std_rms.values)
    e = np.sqrt(df.sse_full.values[v] / 65 / 65)
    dis["psi_rms"] = {"pearson_rmse_vs_memstd": round(float(np.corrcoef(e, df.psi_std_rms.values[v])[0, 1]), 4),
                      "rms_blend_err": round(float(np.sqrt(np.mean(e ** 2))), 6),
                      "rms_member_std": round(float(np.sqrt(np.mean(df.psi_std_rms.values[v] ** 2))), 6)}
    v = np.isfinite(df.D_LCFS.values) & np.isfinite(df.D_memstd.values)
    dis["D_LCFS"] = {"pearson_D_vs_memstd": round(float(np.corrcoef(df.D_LCFS.values[v], df.D_memstd.values[v])[0, 1]), 4),
                     "spearman": round(float(sst.spearmanr(df.D_LCFS.values[v], df.D_memstd.values[v]).correlation), 4),
                     "mean_D_blend": round(float(df.D_LCFS[v].mean()), 6), "mean_D_avg_member": round(float(np.nanmean(mem["mem_d"])), 6)}
    for s in ("q95", "betaN"):
        e = np.abs(errs[s]); sd = df[f"{s}_std"].values
        dis[s] = {"pearson_abs_err_vs_memstd": round(float(np.corrcoef(e, sd)[0, 1]), 4),
                  "rms_blend_err": round(float(np.sqrt(np.mean(e ** 2))), 6), "rms_member_std": round(float(np.sqrt(np.mean(sd ** 2))), 6),
                  "mean_signed_err": round(float(np.mean(errs[s])), 6)}
    rep["disagreement"] = dis

    # ------------------------------------------------------------------ 2. decompositions
    df["frac_dec"], _ = qbins(df.frac.values, 10)
    df["Ip_q"], ip_edges = qbins(np.abs(df.Ip_kA.values), 5)
    df["amp_q"], amp_edges = qbins(df.amp_gt.values, 5)
    df["peak_differs"] = ((df.iz_gt != df.iz_pred) | (df.ir_gt != df.ir_pred)).astype(float)
    df["null_cls"] = df.null_gt.fillna(-1)
    df["xlo_sign"] = np.where(df.null_gt == 1, "lower", np.where(df.null_gt == 2, "upper", np.where(df.null_gt == 3, "double", "none")))
    rep["edges"] = {"Ip_kA_quintiles": [round(float(x), 1) for x in ip_edges], "amp_quintiles": [round(float(x), 4) for x in amp_edges]}
    rep["class_counts"] = {"null_gt": {NULL_NAMES.get(int(k), str(k)): int(v) for k, v in df.null_gt.value_counts().items()},
                           "limited_gt": {str(int(k)): int(v) for k, v in df.limited_gt.value_counts().items()},
                           "null_pred": {NULL_NAMES.get(int(k), str(k)): int(v) for k, v in df.null_pred.value_counts().items()},
                           "null_pred_ne_gt": int((df.null_pred != df.null_gt).sum()),
                           "peak_cell_differs": int(df.peak_differs.sum())}
    # D_LCFS is a mean of shot means: weight frame k by 1/T_k
    dweight = 1.0 / df["T"].values
    targets = {s: (errs[s] ** 2, np.ones(N), valid[s]) for s in CONS + ["q95", "betaN"]}
    targets["D_LCFS"] = (df.D_LCFS.values, dweight, np.isfinite(df.D_LCFS.values))
    targets["psi_sse"] = (df.sse_full.values, np.ones(N), np.ones(N, bool))
    keys = {"frac_decile": df.frac_dec.values, "Ip_quintile": df.Ip_q.values, "amp_quintile": df.amp_q.values,
            "null_gt": df.null_gt.values, "limited_gt": df.limited_gt.values, "peak_cell_differs": df.peak_differs.values,
            "null_pred_ne_gt": (df.null_pred != df.null_gt).astype(float).values}
    dec = {}
    for tname, (e2, w, v) in targets.items():
        dec[tname] = {}
        for kname, key in keys.items():
            labels = NULL_NAMES if kname in ("null_gt",) else None
            dec[tname][kname] = group_shares(np.where(v, e2, np.nan), w, key, labels)
        # by shot: top-10 share
        sh = pd.DataFrame({"e": np.where(v, e2 * w, np.nan), "shot": df.shot_index.values, "vi": df.validation_index.values})
        g = sh.groupby(["vi", "shot"]).e.sum().sort_values(ascending=False)
        tot = g.sum()
        dec[tname]["by_shot_top10"] = [{"validation_index": int(vi), "shot_index": int(si), "ss_share": round(float(x / tot), 4),
                                        "n_frames": int((df.validation_index == vi).sum())} for (vi, si), x in g.head(10).items()]
        dec[tname]["top10_shots_share"] = round(float(g.head(10).sum() / tot), 4)
        dec[tname]["top1pct_frames_share"] = round(float(np.sort(np.nan_to_num(sh.e.values))[::-1][:max(1, N // 100)].sum() / tot), 4)
        dec[tname]["top10pct_frames_share"] = round(float(np.sort(np.nan_to_num(sh.e.values))[::-1][:N // 10].sum() / tot), 4)
    rep["decomposition"] = dec

    # ------------------------------------------------------------------ 3. bias structure (OLS)
    bias = {}
    Zmax_err = (df.Zmax_pred - df.Zmax_gt).values; Zmin_err = (df.Zmin_pred - df.Zmin_gt).values
    Rmax_err = (df.Rmax_pred - df.Rmax_gt).values; Rmin_err = (df.Rmin_pred - df.Rmin_gt).values
    absIp = np.abs(df.Ip_kA.values); tms = df.time_ms.values; cov = df.coverage.values
    for s, e, g in [(s, errs[s], df[f"{s}_gt"].values) for s in CONS] + \
                   [("Zmax", Zmax_err, df.Zmax_gt.values), ("Zmin", Zmin_err, df.Zmin_gt.values),
                    ("Rmax", Rmax_err, df.Rmax_gt.values), ("Rmin", Rmin_err, df.Rmin_gt.values),
                    ("q95", errs["q95"], df.q95_gt.values), ("betaN", errs["betaN"], df.betaN_gt.values)]:
        bias[s] = {"mean_err": round(float(np.nanmean(e)), 6), "rmse": round(float(np.sqrt(np.nanmean(e ** 2))), 6),
                   "median_err": round(float(np.nanmedian(e)), 6),
                   "vs_gt": ols(e, [g], ["gt"]),
                   "vs_gt_inputs": ols(e, [g, absIp, tms, cov], ["gt", "absIp_kA", "time_ms", "coverage"])}
        if s in base and "ss_tot" in base[s]:
            bias[s]["r2_if_linear_bias_removed"] = round(1 - bias[s]["vs_gt"]["ss_after"] / base[s]["ss_tot"], 6)
            bias[s]["r2_if_gt_inputs_bias_removed"] = round(1 - bias[s]["vs_gt_inputs"]["ss_after"] / base[s]["ss_tot"], 6)
            bias[s]["r2_if_mean_removed"] = round(1 - float(np.nansum((e - np.nanmean(e)) ** 2)) / base[s]["ss_tot"], 6)
    # kappa linearisation: dk = (dZmax - dZmin)/w - k (dRmax - dRmin)/w
    w = (df.Rmax_gt - df.Rmin_gt).values; kg = df.kappa_gt.values
    tZmax = Zmax_err / w; tZmin = -Zmin_err / w; tR = -kg * (Rmax_err - Rmin_err) / w
    kerr = errs["kappa"]
    m = np.isfinite(kerr) & np.isfinite(tZmax) & np.isfinite(tZmin) & np.isfinite(tR)
    lin = tZmax + tZmin + tR
    ss = float(np.sum(kerr[m] ** 2))
    bias["kappa_decomposition"] = {"linearisation_r2": round(1 - float(np.sum((kerr[m] - lin[m]) ** 2)) / ss, 4),
                                   "share_Zmax_term": round(float(np.sum(tZmax[m] * kerr[m])) / ss, 4),
                                   "share_Zmin_term": round(float(np.sum(tZmin[m] * kerr[m])) / ss, 4),
                                   "share_width_term": round(float(np.sum(tR[m] * kerr[m])) / ss, 4),
                                   "rms_Zmax_err_m": round(float(np.sqrt(np.nanmean(Zmax_err ** 2))), 5),
                                   "rms_Zmin_err_m": round(float(np.sqrt(np.nanmean(Zmin_err ** 2))), 5),
                                   "rms_Rmax_err_m": round(float(np.sqrt(np.nanmean(Rmax_err ** 2))), 5),
                                   "rms_Rmin_err_m": round(float(np.sqrt(np.nanmean(Rmin_err ** 2))), 5),
                                   "note": "shares are cov(term, kappa_err)/SS_res(kappa); they sum to the linearisation R2"}
    # volume: 2*pi*R*A ; linearise via ln
    vg = df.volume_gt.values; vp = df.volume_pred.values
    bias["volume_rel_err"] = {"mean_rel": round(float(np.nanmean(vp / vg - 1)), 5), "rms_rel": round(float(np.sqrt(np.nanmean((vp / vg - 1) ** 2))), 5)}
    # axis: subpixel vs cell
    ra = errs["R_axis"]; za = errs["Z_axis"]; pdiff = df.peak_differs.values > 0
    bias["axis"] = {"rms_R_axis_m": round(float(np.sqrt(np.nanmean(ra ** 2))), 5), "rms_Z_axis_m": round(float(np.sqrt(np.nanmean(za ** 2))), 5),
                    "frac_peak_cell_differs": round(float(pdiff.mean()), 4),
                    "rms_R_axis_same_cell": round(float(np.sqrt(np.nanmean(ra[~pdiff] ** 2))), 5),
                    "rms_R_axis_diff_cell": round(float(np.sqrt(np.nanmean(ra[pdiff] ** 2))), 5),
                    "ss_share_R_axis_diff_cell": round(float(np.nansum(ra[pdiff] ** 2) / np.nansum(ra ** 2)), 4),
                    "ss_share_Z_axis_diff_cell": round(float(np.nansum(za[pdiff] ** 2) / np.nansum(za ** 2)), 4),
                    "cell_dR_m": 0.026563, "cell_dZ_m": 0.05,
                    "corr_R_axis_err_vs_err_at_axis": round(float(pd.Series(ra).corr(df.err_at_axis)), 4),
                    "corr_R_axis_err_vs_std_at_axis": round(float(pd.Series(np.abs(ra)).corr(df.std_at_axis)), 4)}
    rep["bias"] = bias

    # ------------------------------------------------------------------ 4. x-point region
    xp = {}
    lsn = (df.null_gt == 1).values
    for side, sel_name in (("lo", "lower-SN(gt)"), ("hi", "upper-SN(gt)")):
        sel = (df.null_gt == (1 if side == "lo" else 2)).values
        sel &= np.isfinite(df[f"x{side}_R_pred"].values) & np.isfinite(df[f"x{side}_R_gt"].values)
        dRx = (df[f"x{side}_R_pred"] - df[f"x{side}_R_gt"]).values[sel]
        dZx = (df[f"x{side}_Z_pred"] - df[f"x{side}_Z_gt"]).values[sel]
        Zext_err = (Zmin_err if side == "lo" else Zmax_err)[sel]
        Zext_gt_off = ((df.Zmin_gt - df.xlo_Z_gt) if side == "lo" else (df.Zmax_gt - df.xhi_Z_gt)).values[sel]
        Zext_pr_off = ((df.Zmin_pred - df.xlo_Z_pred) if side == "lo" else (df.Zmax_pred - df.xhi_Z_pred)).values[sel]
        lev = (df[f"err_at_x{side}"].values / df.amp_gt.values)[sel] if f"err_at_x{side}" in df else np.full(sel.sum(), np.nan)
        m = np.isfinite(dZx) & np.isfinite(Zext_err)
        o = ols(Zext_err[m], [dZx[m]], ["dZ_xpoint"])
        xp[sel_name] = {"n": int(sel.sum()),
                        "xpoint_dR_mean_m": round(float(np.nanmean(dRx)), 5), "xpoint_dR_rms_m": round(float(np.sqrt(np.nanmean(dRx ** 2))), 5),
                        "xpoint_dZ_mean_m": round(float(np.nanmean(dZx)), 5), "xpoint_dZ_rms_m": round(float(np.sqrt(np.nanmean(dZx ** 2))), 5),
                        "level_err_at_x_gt_over_amp_mean": round(float(np.nanmean(lev)), 6),
                        "level_err_at_x_gt_over_amp_rms": round(float(np.sqrt(np.nanmean(lev ** 2))), 6),
                        "Zext_err_rms_m": round(float(np.sqrt(np.nanmean(Zext_err ** 2))), 5),
                        "Zext_err_mean_m": round(float(np.nanmean(Zext_err)), 5),
                        "corr_Zext_err_vs_dZ_xpoint": round(float(np.corrcoef(Zext_err[m], dZx[m])[0, 1]), 4),
                        "frac_Zext_SS_explained_by_dZ_xpoint": o["frac_ss_removed"], "ols_slope": o["coef"]["dZ_xpoint"],
                        "gt_Zext_minus_xZ_mean_m": round(float(np.nanmean(Zext_gt_off)), 5),
                        "gt_Zext_minus_xZ_rms_m": round(float(np.sqrt(np.nanmean(Zext_gt_off ** 2))), 5),
                        "gt_Zext_minus_xZ_p10_p50_p90": [round(float(x), 4) for x in np.nanquantile(Zext_gt_off, [.1, .5, .9])],
                        "pred_Zext_minus_xZ_rms_m": round(float(np.sqrt(np.nanmean(Zext_pr_off ** 2))), 5),
                        "rounding_term_rms_m": round(float(np.sqrt(np.nanmean((Zext_pr_off - Zext_gt_off) ** 2))), 5),
                        "note": "Zext_err = (dZ_xpoint) + (rounding: (Zext-xZ)_pred - (Zext-xZ)_gt); rounding = how far the marching-squares LCFS stops short of the saddle"}
    # Zmax on lower-SN frames (smooth top): what drives it?
    sel = lsn & np.isfinite(Zmax_err)
    xp["Zmax_on_lowerSN_frames"] = {"n": int(sel.sum()), "rms_m": round(float(np.sqrt(np.mean(Zmax_err[sel] ** 2))), 5),
                                    "mean_m": round(float(np.mean(Zmax_err[sel])), 5),
                                    "share_of_total_Zmax_SS": round(float(np.sum(Zmax_err[sel] ** 2) / np.nansum(Zmax_err ** 2)), 4),
                                    "corr_with_upper_saddle_dlev_gt": round(float(pd.Series(Zmax_err[sel]).corr(df.xhi_dlev_gt[sel])), 4),
                                    "corr_with_amp_err": round(float(pd.Series(Zmax_err[sel]).corr((df.amp_pred - df.amp_gt)[sel])), 4),
                                    "corr_with_bdry_level_err": round(float(pd.Series(Zmax_err[sel]).corr((df.psi_bdry_pred - df.psi_bdry_gt)[sel])), 4)}
    sel = lsn & np.isfinite(Zmin_err)
    xp["Zmin_on_lowerSN_frames"] = {"n": int(sel.sum()), "rms_m": round(float(np.sqrt(np.mean(Zmin_err[sel] ** 2))), 5),
                                    "share_of_total_Zmin_SS": round(float(np.sum(Zmin_err[sel] ** 2) / np.nansum(Zmin_err ** 2)), 4)}
    # kappa SS by null class
    rep["xpoint"] = xp

    # ------------------------------------------------------------------ 5. li anatomy
    li = {}
    le = errs["li"]; v = valid["li"]
    top = np.argsort(-np.abs(np.where(v, le, 0)))[:200]
    gp = prof["gt_prof"]; pp = prof["pr_prof"]; ppg = prof["pr_prof_gtc"]
    def prof_ratio(idx):
        r = np.nanmean(pp[idx] / gp[idx], axis=0); rg = np.nanmean(ppg[idx] / gp[idx], axis=0)
        return {"pred_over_gt_Bp2_by_s_bin": [round(float(x), 4) for x in r],
                "pred_on_gt_contour_over_gt": [round(float(x), 4) for x in rg]}
    li["top200_profile"] = prof_ratio(top); li["all_profile"] = prof_ratio(np.where(v)[0])
    li["top200_share_of_SS"] = round(float(np.sum(le[top] ** 2) / np.nansum(le[v] ** 2)), 4)
    li["top200_mean_signed_err"] = round(float(np.mean(le[top])), 5)
    li["top200_null_gt_counts"] = {NULL_NAMES.get(int(k), str(k)): int(c) for k, c in pd.Series(df.null_gt.values[top]).value_counts().items()}
    li["top200_frac_deciles"] = [int(c) for c in np.bincount(df.frac_dec.values[top].astype(int), minlength=10)]
    # log decomposition: ln li = ln Bp2V - 2 ln (G/lp)
    dln = np.log(df.li_pred / df.li_gt).values
    dnum = np.log(df.Bp2V_pred / df.Bp2V_gt).values; dden = -2 * np.log(df.GoverLp_pred / df.GoverLp_gt).values
    m = v & np.isfinite(dln) & np.isfinite(dnum) & np.isfinite(dden)
    ss = float(np.sum(dln[m] ** 2))
    li["log_decomposition"] = {"share_numerator_<Bp2>_V": round(float(np.sum(dnum[m] * dln[m])) / ss, 4),
                               "share_normalisation_(mu0Ip/lp)^2": round(float(np.sum(dden[m] * dln[m])) / ss, 4),
                               "rms_dln_li": round(float(np.sqrt(ss / m.sum())), 5),
                               "rms_dln_num": round(float(np.sqrt(np.mean(dnum[m] ** 2))), 5),
                               "rms_dln_den": round(float(np.sqrt(np.mean(dden[m] ** 2))), 5),
                               "corr_num_den": round(float(np.corrcoef(dnum[m], dden[m])[0, 1]), 4)}
    # contour vs field: li(pred psi, GT contour)
    lpg = df.li_pred_gtcontour.values
    m = v & np.isfinite(lpg)
    li["contour_vs_field"] = {"ss_res_li_blend": float(np.sum(le[m] ** 2)),
                              "ss_res_li_pred_on_gt_contour": float(np.sum((lpg[m] - df.li_gt.values[m]) ** 2)),
                              "r2_li_pred_on_gt_contour": round(1 - float(np.sum((lpg[m] - df.li_gt.values[m]) ** 2)) / base["li"]["ss_tot"], 6),
                              "note": "li computed with the official internal_inductance() on the blend psi but the GT contour: the remainder is pure field (gradient) error"}
    li["mean_signed_err"] = round(float(np.nanmean(le[v])), 5)
    li["corr_li_err_vs_kappa_err"] = round(float(pd.Series(le).corr(pd.Series(errs["kappa"]))), 4)
    li["corr_li_err_vs_volume_relerr"] = round(float(pd.Series(le).corr(pd.Series(vp / vg - 1))), 4)
    li["corr_li_err_vs_j_rms_diff"] = round(float(pd.Series(np.abs(le)).corr(df.j_rms_diff)), 4)
    rep["li"] = li

    # ------------------------------------------------------------------ 6. current density
    jd = {}
    m = np.isfinite(df.j_rms_diff.values)
    jd["rms_diff_over_rms_gt_median"] = round(float(np.nanmedian(df.j_rms_diff / df.j_rms_gt)), 4)
    jd["rms_diff_over_rms_gt_mean"] = round(float(np.nanmean(df.j_rms_diff / df.j_rms_gt)), 4)
    jd["hf_fraction_gt_median"] = float(np.nanmedian(df.j_hf_gt)); jd["hf_fraction_pred_median"] = float(np.nanmedian(df.j_hf_pred))
    jd["hf_fraction_gt_mean"] = float(np.nanmean(df.j_hf_gt)); jd["hf_fraction_pred_mean"] = float(np.nanmean(df.j_hf_pred))
    jd["hf_fraction_of_residual_median"] = float(np.nanmedian(df.j_hf_diff))
    jd["frac_frames_pred_noisier"] = round(float(np.nanmean(df.j_hf_pred > df.j_hf_gt)), 4)
    jd["Ip_from_j_gt_over_Ip_meas_median"] = round(float(np.nanmedian(np.abs(df.Ip_from_j_gt_kA) / np.abs(df.Ip_kA))), 4)
    jd["Ip_from_j_pred_over_gt_median"] = round(float(np.nanmedian(df.Ip_from_j_pred_kA / df.Ip_from_j_gt_kA)), 5)
    jd["Ip_from_j_pred_over_gt_mean"] = round(float(np.nanmean(df.Ip_from_j_pred_kA / df.Ip_from_j_gt_kA)), 5)
    jd["corr_j_rms_diff_vs_psi_std"] = round(float(df.j_rms_diff.corr(df.psi_std_rms)), 4)
    jd["corr_j_rms_diff_vs_kappa_abs_err"] = round(float(df.j_rms_diff.corr(pd.Series(np.abs(errs["kappa"])))), 4)
    rep["current_density"] = jd

    # ------------------------------------------------------------------ 7. maps + amplitude/gauge
    me = maps["mean_err"]; rm = maps["rms_err"]; ins = maps["inside_frac"]
    amp_err = (df.amp_pred - df.amp_gt).values; bd_err = (df.psi_bdry_pred - df.psi_bdry_gt).values; ax_err = (df.psi_axis_pred - df.psi_axis_gt).values
    rep["gauge"] = {"mean_err_map_abs_max": float(np.abs(me).max()), "mean_err_map_rms": float(np.sqrt(np.mean(me ** 2))),
                    "rms_err_map_mean": float(np.mean(rm)), "rms_err_inside_mean": float(np.sum(rm * ins) / np.sum(ins)),
                    "rms_err_outside_mean": float(np.sum(rm * (1 - ins)) / np.sum(1 - ins)),
                    "mean_err_map_argmax_RZ_index": [int(x) for x in np.unravel_index(int(np.argmax(np.abs(me))), me.shape)],
                    "rms_err_map_argmax_RZ_index": [int(x) for x in np.unravel_index(int(np.argmax(rm)), rm.shape)],
                    "psi_axis_err_mean": round(float(np.nanmean(ax_err)), 6), "psi_axis_err_rms": round(float(np.sqrt(np.nanmean(ax_err ** 2))), 6),
                    "psi_bdry_err_mean": round(float(np.nanmean(bd_err)), 6), "psi_bdry_err_rms": round(float(np.sqrt(np.nanmean(bd_err ** 2))), 6),
                    "amp_err_mean": round(float(np.nanmean(amp_err)), 6), "amp_err_rms": round(float(np.sqrt(np.nanmean(amp_err ** 2))), 6),
                    "amp_rel_err_mean": round(float(np.nanmean(amp_err / df.amp_gt.values)), 6),
                    "corr_amp_err_vs_amp_gt": round(float(pd.Series(amp_err).corr(df.amp_gt)), 4),
                    "corr_amp_relerr_vs_li_err": round(float(pd.Series(amp_err / df.amp_gt.values).corr(pd.Series(errs["li"]))), 4),
                    "corr_amp_relerr_vs_volume_relerr": round(float(pd.Series(amp_err / df.amp_gt.values).corr(pd.Series(vp / vg - 1))), 4)}
    # ------------------------------------------------------------------ 8. hard frames + rounding floor
    hard = {}
    pop = {"absIp_kA": float(np.nanmedian(absIp)), "frac": float(np.nanmedian(df.frac)), "amp_gt": float(np.nanmedian(df.amp_gt)),
           "li_gt": float(np.nanmedian(df.li_gt)), "kappa_gt": float(np.nanmedian(df.kappa_gt)), "volume_gt": float(np.nanmedian(df.volume_gt)),
           "psi_std_rms": float(np.nanmedian(df.psi_std_rms)), "D_LCFS": float(np.nanmedian(df.D_LCFS)),
           "time_ms": float(np.nanmedian(df.time_ms)), "coverage": float(np.nanmean(df.coverage))}
    hard["population_medians"] = pop
    for s in CONS + ["q95", "betaN", "D_LCFS", "psi"]:
        if s == "D_LCFS":
            e2 = np.nan_to_num(df.D_LCFS.values)
        elif s == "psi":
            e2 = df.sse_full.values
        else:
            e2 = np.where(valid[s], errs[s] ** 2, 0.0)
        k = max(1, N // 100)
        top = np.argsort(-e2)[:k]
        sub = df.iloc[top]
        hard[s] = {"n_top1pct": int(k), "ss_share": round(float(e2[top].sum() / e2.sum()), 4),
                   "median_absIp_kA": float(np.nanmedian(np.abs(sub.Ip_kA))), "median_frac": float(np.nanmedian(sub.frac)),
                   "frac_in_last_decile": round(float(np.mean(sub.frac >= 0.9)), 3), "frac_in_first_decile": round(float(np.mean(sub.frac <= 0.1)), 3),
                   "median_amp_gt": float(np.nanmedian(sub.amp_gt)), "median_li_gt": float(np.nanmedian(sub.li_gt)),
                   "median_kappa_gt": float(np.nanmedian(sub.kappa_gt)), "median_volume_gt": float(np.nanmedian(sub.volume_gt)),
                   "median_psi_std_rms": float(np.nanmedian(sub.psi_std_rms)), "median_D_LCFS": float(np.nanmedian(sub.D_LCFS)),
                   "median_time_ms": float(np.nanmedian(sub.time_ms)), "mean_coverage": float(np.nanmean(sub.coverage)),
                   "null_gt_counts": {NULL_NAMES.get(int(x), str(x)): int(c) for x, c in sub.null_gt.value_counts().items()},
                   "n_distinct_shots": int(sub.validation_index.nunique()),
                   "frac_peak_cell_differs": round(float(sub.peak_differs.mean()), 3),
                   "frac_null_pred_ne_gt": round(float((sub.null_pred != sub.null_gt).mean()), 3),
                   "median_abs_xlo_dR": float(np.nanmedian(np.abs(sub.xlo_R_pred - sub.xlo_R_gt))),
                   "median_abs_xlo_dZ": float(np.nanmedian(np.abs(sub.xlo_Z_pred - sub.xlo_Z_gt))),
                   "mean_signed_err": (float(np.nanmean(errs[s][top])) if s in errs else None)}
    # low-Ip / small-plasma slice
    lowip = absIp < np.nanquantile(absIp, 0.1)
    hard["lowest_Ip_decile"] = {"Ip_max_kA": float(np.nanquantile(absIp, 0.1)),
                                **{s: {"ss_share": round(float(np.nansum(np.where(valid[s] & lowip, errs[s] ** 2, 0)) / np.nansum(np.where(valid[s], errs[s] ** 2, 0))), 4)} for s in CONS}}
    # rounding floor: lower-SN frames whose predicted saddle is within 3 mm (Z) / 5 mm (R) of GT saddle
    sel = lsn & (np.abs(df.xlo_Z_pred - df.xlo_Z_gt).values < 0.003) & (np.abs(df.xlo_R_pred - df.xlo_R_gt).values < 0.005) & np.isfinite(Zmin_err)
    hard["rounding_floor_lowerSN"] = {"n_frames_saddle_within_3mm": int(sel.sum()), "frac_of_lowerSN": round(float(sel.sum() / max(1, lsn.sum())), 4),
                                      "Zmin_err_rms_m": round(float(np.sqrt(np.mean(Zmin_err[sel] ** 2))), 5),
                                      "Zmin_err_rms_all_lowerSN_m": round(float(np.sqrt(np.nanmean(Zmin_err[lsn] ** 2))), 5),
                                      "kappa_err_rms_these": round(float(np.sqrt(np.nanmean(errs["kappa"][sel] ** 2))), 5),
                                      "kappa_err_rms_all_lowerSN": round(float(np.sqrt(np.nanmean(errs["kappa"][lsn] ** 2))), 5),
                                      "kappa_r2_if_all_frames_had_this_rms": round(1 - float(np.nanmean(errs["kappa"][sel] ** 2)) * base["kappa"]["n"] / base["kappa"]["ss_tot"], 5)}
    off_gt = (df.Zmin_gt - df.xlo_Z_gt).values
    m = lsn & np.isfinite(off_gt) & np.isfinite(Zmin_err)
    hard["Zmin_err_vs_gt_rounding_offset_corr"] = round(float(np.corrcoef(off_gt[m], Zmin_err[m])[0, 1]), 4)
    hard["Zmin_err_rms_when_gt_offset_gt_2cm"] = round(float(np.sqrt(np.mean(Zmin_err[m & (np.abs(off_gt) > 0.02)] ** 2))), 5)
    hard["Zmin_err_rms_when_gt_offset_lt_1cm"] = round(float(np.sqrt(np.mean(Zmin_err[m & (np.abs(off_gt) < 0.01)] ** 2))), 5)
    rep["hard_frames"] = hard

    rep["ceilings"] = {k: {kk: v[kk] for kk in ("S", "delta_S_vs_blend", "r2_psi", "r2_qb", "dlcfs", "consistency", "r2_cons_each")} for k, v in ceil.items()}

    df.to_parquet(OUT / "frames_enriched.parquet", index=False)
    (OUT / "report.json").write_text(json.dumps(rep, indent=1, default=float) + "\n")
    print(json.dumps(rep["baseline"], indent=1))
    return rep


if __name__ == "__main__":
    main()
