#!/usr/bin/env python3
"""Fidelity of the synthetic free-boundary generator on DIII-D, where EFIT labels exist.

For N DIII-D TRAIN frames (advanced_v1 caches: real coil currents, Ip, TF; EFIT psi label) the MAST generator
physics (fusion.c2_synthetic via fusion.c2_synthetic_d3d) is run on the DIII-D geometry in several arms and every
arm's flux map is scored against the EFIT label with the OFFICIAL scorer functions (starter/fusion_scoring):

  sampled_centroid  MAST-v1 sampled profile + MAST-v1 position prior scaled to the DIII-D envelope centroid,
                    beta0 balanced so the real field holds the plasma at r_target      (= what the generator does)
  sampled_efitpos   same sampled profile, current-centroid target = EFIT current centroid (label-informed position)
  fitted_efitpos    FreeGS profile least-squares fitted to the EFIT j, EFIT centroid, radial+vertical control free
  fitted_vertical   fitted profile, vertical control only: radial position left to the real vacuum field
  fitted_bzfix      fitted_efitpos + a per-frame uniform vertical-field correction fitted on the exterior residual
  fitted_multipole  fitted_efitpos + the full exterior multipole correction (uniform B_Z + quadrupole + odd terms)
  oracle_j          EFIT current density placed in the real coil field (linear; no profile, no control)

Outputs: <output>/frames.csv (one row per frame x arm), <output>/report.json (per-arm medians / p90, pooled R2,
bias statistics, dependence on control effort and decay index, multipole regression), <output>/config.json.
Labels: DIII-D train only (diagnostic). No MAST data is touched.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "starter" / "fusion_scoring"))
from fusion import c2_fields_data as cd  # noqa: E402
from fusion import c2_synthetic as cs  # noqa: E402
from fusion import c2_synthetic_d3d as cd3  # noqa: E402
from fusion import physics_fields as pf  # noqa: E402
from contour import symmetric_hausdorff  # noqa: E402
from derive import derive_frame, magnetic_axis  # noqa: E402
from lcfs import extract_lcfs, major_radius  # noqa: E402

ARMS = ["sampled_centroid", "sampled_efitpos", "fitted_efitpos", "fitted_vertical", "fitted_bzfix", "fitted_multipole", "oracle_j",
        "fitted_rawcoil", "oracle_j_rawcoil"]
# extra arms (--arms): sampled_vertical (MAST-v1 profile, beta0 fixed, vertical control only: radial position from the real field),
# sampledD3D_efitpos / sampledD3D_centroid (profile sampled from DIII-D-FITTED ranges, q10-q90 of the fitted parameters)
D3D_FIT_RANGES = {"beta0": (0.03, 0.95), "alpha_p": (0.4, 6.0), "gamma_p": (0.3, 1.1), "alpha_f": (0.78, 2.5), "gamma_f": (0.78, 5.2)}
EXTRA_ARMS = ["sampled_vertical", "sampledD3D_efitpos", "sampledD3D_centroid"]
ARM_DEFINITIONS = {
    "sampled_centroid": "MAST-v1 sampled profile (MAST_V1_RANGES) + MAST-v1 position prior scaled to the DIII-D envelope centroid; beta0 balanced (dI5 -> 0); calibrated E-coil",
    "sampled_efitpos": "same sampled profile, current-centroid target = EFIT current centroid; beta0 balanced",
    "fitted_efitpos": "FreeGS family least-squares fitted to the EFIT j (5 params fixed), EFIT centroid, radial (F6 sym) + vertical (F7 antisym) control free",
    "fitted_vertical": "fitted profile, vertical control only; radial position from the real field",
    "fitted_bzfix": "fitted_efitpos + per-frame uniform B_Z correction fitted on the exterior residual of fitted_efitpos",
    "fitted_multipole": "fitted_efitpos + full exterior multipole correction (uniform B_Z, quadrupole, Z, R^2 Z)",
    "oracle_j": "EFIT j (renormalised to measured Ip) in the calibrated coil field; linear, no profile, no control",
    "fitted_rawcoil": "fitted_efitpos with the RAW physics_fields coil model (ECOILA 137-turn table, shape factor 1)",
    "oracle_j_rawcoil": "oracle_j with the RAW coil model",
    "sampled_vertical": "MAST-v1 sampled profile, beta0 fixed, vertical control only (radial position from the real field)",
    "sampledD3D_efitpos": "profile sampled from D3D_FIT_RANGES (q10-q90 of the DIII-D fitted parameters), EFIT centroid, beta0 balanced",
    "sampledD3D_centroid": "profile sampled from D3D_FIT_RANGES, MAST-v1 centroid position prior, beta0 balanced",
    # ---- v9 arms (position from force balance, label-free beta_p from the frame's own Thomson data) ---------------- #
    "balance_v9": "MAST-v1 sampled profile; radial position FREE (vertical F7 control only, dI5 = 0); beta0 secant until the closure beta_p"
                  " equals the LABEL-FREE target = Thomson proxy beta_p,e (fusion.c2_synthetic.thomson_beta_p_proxy, layout d3d) x ratio U(1.5,2.5)"
                  " x lognormal(0.3) jitter; z_target = MAST-v1 Z prior; current seeded at r_init (label-free)",
    "balance_edge_v9": "balance_v9 + v9 edge/pedestal current term (edge_amp ~ U(0,0.3), edge_w ~ U(0.03,0.12))",
    "balance_nojit_v9": "balance_v9 without the lognormal jitter (proxy x ratio only): isolates the jitter's cost",
    "balance_labelbp_v9": "position from balance with the LABEL beta_p (closure beta_p of the fitted profile): label-informed ceiling of the balance mode",
    "balance_edge_labelbp_v9": "balance_labelbp_v9 + edge term",
    "balance_edge_relax_v9": "balance_edge_v9 with Picard relaxation 0.4 and 160 iterations (convergence test of the edge term)",
    # ---- iteration 3: position from the (simulated) MIDPLANE Thomson chord, pressure from force balance ------------ #
    "chord_axis_v9": "MAST-v1 sampled profile; centroid pinned (dI5 -> 0 by the beta0 balance as in v1-v3) at a radius iterated until the"
                     " pressure-core centre R_axis_est of the equilibrium's own midplane chord equals the one measured on the SIMULATED"
                     " DIII-D midplane chord (measured DIII-D Thomson p_e mapped to normalised flux with the label, re-sampled on Z = 0 at"
                     " 1 cm, 5 % noise: the label is used ONLY to emulate MAST's horizontal chord geometry); seed R_axis_est - 0.05 m;"
                     " at most --chord-steps warm-started re-solves; z_target = MAST-v1 Z prior; no label position, no dI5 fudge",
    "chord_rgeo_v9": "chord_axis_v9 but matching the LCFS midplane centre Rgeo_est (foot-extrapolated hot extent) instead of the core centre",
    "chord_axis_edge_v9": "chord_axis_v9 + v9 edge/pedestal current term (edge_amp ~ U(0,0.3), edge_w ~ U(0.03,0.12))",
    "chord_axis_noise_v9": "chord_axis_v9 with the measured target perturbed by N(0, --chord-noise m): sensitivity to the chord estimate",
    "chord_rgeo_edge_v9": "chord_rgeo_v9 + edge term",
    "sampled_centroid_zlabel": "sampled_centroid with the LABEL centroid Z (MAST-v1 radius prior kept): isolates the radial-position effect on DIII-D",
}
# Vertical position of the chord arms: --chord-z 'label' (default) pins Z at the LABEL current centroid exactly as sampled_efitpos does
# (DIII-D plasmas sit up to 0.1 m off the MAST-v1 Z prior and the vertical DIII-D Thomson chord covers Z > 0.03 m only, so no label-free
# Z estimate exists on DIII-D; MAST plasmas sit at Z ~ 0 where the generator's Z prior / double-null control applies), or 'prior'.
V9_ARMS = ["balance_v9", "balance_edge_v9", "balance_nojit_v9", "balance_labelbp_v9", "balance_edge_labelbp_v9", "balance_edge_relax_v9"]
CHORD_ARMS = ["chord_axis_v9", "chord_rgeo_v9", "chord_axis_edge_v9", "chord_axis_noise_v9", "chord_rgeo_edge_v9"]
V9_ARMS = V9_ARMS + CHORD_ARMS
EXTRA_ARMS = EXTRA_ARMS + ["sampled_centroid_zlabel"]
D3D_CHORD_R_BOUNDS = (1.30, 2.10)      # admissible pinned centroid radius on DIII-D (m)


def simulated_midplane_chord(shot, local, phi_gt, solver, rng, noise=0.05, R_range=(0.95, 2.45), dR=0.01, Z0=0.0, n_bins=20):
    """MAST-like horizontal Thomson chord for a DIII-D frame (validation-only emulation, documented label use).

    The MEASURED DIII-D Thomson p_e (vertical core chord R = 1.94 m + tangential chord) is mapped to the normalised flux
    s of the EFIT label at the measurement points (the ONLY label use here: the flux-surface mapping that MAST's chord
    geometry does not need), binned to a median profile p_e(s) with p_e(1) = 0, and re-sampled along Z = Z0 at ``dR``
    spacing inside the label LCFS with multiplicative Gaussian noise. The generator then receives only
    (chord_R, chord_Z, pe, mask), exactly what it gets from MAST's real chord."""
    _, surface = solver.label_current(phi_gt)
    C, _, axis_val, edge, _ = surface
    delta = axis_val - edge
    R, Z = shot.chord_R.astype(np.float64), shot.chord_Z.astype(np.float64)
    pe = shot.pe[local].astype(np.float64)
    m = shot.pe_mask[local] & np.isfinite(R) & np.isfinite(Z) & np.isfinite(pe)
    psi_c = cs._bilinear(phi_gt, solver.R, solver.Z, R[m], Z[m])
    s = (axis_val - psi_c) / delta
    ins = cs._points_in_polygon(R[m], Z[m], C)
    keep = ins & (s >= 0.0) & (s <= 1.0) & (pe[m] > 0)
    s, p = s[keep], pe[m][keep]
    if len(s) < 8:
        raise ValueError("chord_sim_too_few_points")
    centres = (np.arange(n_bins) + 0.5) / n_bins
    idx = np.minimum((s * n_bins).astype(int), n_bins - 1)
    prof = np.array([np.median(p[idx == i]) if np.any(idx == i) else np.nan for i in range(n_bins)])
    ok = np.isfinite(prof)
    if ok.sum() < 3:
        raise ValueError("chord_sim_too_few_bins")
    prof = np.interp(centres, centres[ok], prof[ok])
    sg = np.concatenate([[0.0], centres, [1.0]])
    pg = np.concatenate([[prof[0]], prof, [0.0]])
    Rm = np.arange(R_range[0], R_range[1] + 1e-9, dR)
    Zm = np.full_like(Rm, Z0)
    psi_m = cs._bilinear(phi_gt, solver.R, solver.Z, Rm, Zm)
    s_m = np.clip((axis_val - psi_m) / delta, 0.0, 1.5)
    in_m = cs._points_in_polygon(Rm, Zm, C)
    pe_m = np.interp(s_m, sg, pg) * in_m
    pe_m = np.maximum(pe_m * (1.0 + noise * rng.standard_normal(len(Rm))), 0.0)
    return Rm.astype(np.float32), Zm.astype(np.float32), pe_m.astype(np.float32), np.ones(len(Rm), bool)
SCALARS = ["R_axis", "Z_axis", "kappa", "tri_top", "tri_bot", "volume", "li"]
_W = {}


def _init(config):
    setup = cd.build_machine_setup("DIII-D")
    _W["setup"] = setup
    raw = cd3.SyntheticD3DSolver(setup, lcfs_iter=config["lcfs_iter"], ecoil_shape_factor=1.0)
    _W["raw"] = raw
    f = float(config.get("ecoil_shape_factor", 1.0))
    _W["solver"] = raw if f == 1.0 else cd3.SyntheticD3DSolver(setup, lcfs_iter=config["lcfs_iter"], ecoil_shape_factor=f, gs=raw.solver)
    _W["config"] = config
    manifest = json.loads((ROOT / config["data_dir"] / "manifest.json").read_text())
    _W["index"] = cd.D3DFeatureIndex(manifest["feature_names"], setup.table_names)
    _W["mask"] = setup.envelope
    _W["maskf"] = setup.envelope.astype(np.float64)


def limiting_xpoint(phi, R, Z, iz, ir, envelope):
    xp = cs.find_xpoints(phi, R, Z, iz, ir, envelope)
    cands = [v for v in (xp["up"], xp["low"]) if v is not None]
    if not cands:
        return (np.nan, np.nan, np.nan, 0)
    best = max(cands, key=lambda v: v[2])
    return (best[0], best[1], best[2], len(cands))


def score_frame(psi_gt, phi_pred, R, Z, mask, maskf, gt_ref):
    """Official-scorer per-frame comparison (DIII-D convention: label axis minimum; prediction = -phi)."""
    psi_pred = -np.asarray(phi_pred, np.float64)
    out = {}
    C_gt, d_gt, rgeo, xp_gt = gt_ref
    C = extract_lcfs(psi_pred, R, Z, "DIII-D", mask, maskf, axis_sign=-1.0)
    out["lcfs_found"] = C is not None
    out["D_LCFS"] = min(1.0, symmetric_hausdorff(C, C_gt) / rgeo) if C is not None else 1.0
    d = derive_frame(psi_pred, R, Z, "DIII-D", mask, maskf, contour=C, axis_sign=-1.0)
    for k in SCALARS:
        out[f"{k}_pred"] = d[k]
        out[f"{k}_err"] = d[k] - d_gt[k]
    if C is not None:
        out.update(Zmax_pred=float(C[:, 1].max()), Zmin_pred=float(C[:, 1].min()), Rmin_pred=float(C[:, 0].min()), Rmax_pred=float(C[:, 0].max()))
    else:
        out.update(Zmax_pred=np.nan, Zmin_pred=np.nan, Rmin_pred=np.nan, Rmax_pred=np.nan)
    out["Zmax_err"] = out["Zmax_pred"] - float(C_gt[:, 1].max())
    out["Zmin_err"] = out["Zmin_pred"] - float(C_gt[:, 1].min())
    out["Rmin_err"] = out["Rmin_pred"] - float(C_gt[:, 0].min())
    out["Rmax_err"] = out["Rmax_pred"] - float(C_gt[:, 0].max())
    # X-point of the prediction (axis-max phi)
    _, _, iz, ir = magnetic_axis(psi_pred, R, Z, "DIII-D", mask, axis_sign=-1.0)
    xr, xz, xpsi, nx = limiting_xpoint(np.asarray(phi_pred, np.float64), R, Z, iz, ir, mask)
    out.update(xpt_R_pred=xr, xpt_Z_pred=xz, xpt_n_pred=nx)
    out["xpt_R_err"] = xr - xp_gt[0]
    out["xpt_Z_err"] = xz - xp_gt[1]
    out["xpt_dist"] = float(np.hypot(out["xpt_R_err"], out["xpt_Z_err"]))
    # flux metrics
    g = np.asarray(psi_gt, np.float64)
    diff = g - psi_pred
    out["psi_ss_res_raw"] = float(np.sum(diff ** 2))
    out["psi_gauge_offset"] = float(diff.mean())
    out["psi_ss_res_gauge"] = float(np.sum((diff - diff.mean()) ** 2))
    ss_tot = float(np.sum((g - g.mean()) ** 2))
    out["r2_psi_frame_raw"] = 1.0 - out["psi_ss_res_raw"] / ss_tot
    out["r2_psi_frame_gauge"] = 1.0 - out["psi_ss_res_gauge"] / ss_tot
    out["psi_rms_gauge"] = float(np.sqrt(out["psi_ss_res_gauge"] / g.size))
    out["ptp_ratio"] = float(np.ptp(psi_pred) / np.ptp(g))
    return out


def calibrate_frame(task):
    """Cheap linear coil-model check of one frame (raw tables): exterior regression + axis vertical-field budget."""
    solver, index = _W["raw"], _W["index"]
    path, local, seed, shot_index = task
    try:
        shot = cd.d3d_shot_inputs(ROOT / path, index, with_labels=True)
        cal = cd3.coil_model_calibration(solver, shot.currents[local].astype(np.float64), float(shot.Ip_kA[local]),
                                         shot.phi[local].astype(np.float64), ip_sign=float(shot.ip_sign))
        cal.update(shot_index=shot_index, frame_index=int(shot.frames[local]), ip_kA=float(shot.Ip_kA[local]),
                   ecoil_kA=float(shot.currents[local][_W["setup"].table_names.index("magnetics_ECOILA")]))
        return cal
    except Exception as exc:  # noqa: BLE001
        return {"shot_index": shot_index, "frame": int(local), "error": f"{type(exc).__name__}:{str(exc)[:100]}"}


def process_frame(task):
    setup, solver, config, index = _W["setup"], _W["solver"], _W["config"], _W["index"]
    raw = _W["raw"]
    cal = config.get("calibration", {}).get(f"{task[3]}_{task[1]}", {})
    mask, maskf = _W["mask"], _W["maskf"]
    R, Z = solver.R, solver.Z
    path, local, seed, shot_index = task
    rng = np.random.default_rng(seed)
    started = time.monotonic()
    rows, info = [], {"shot_index": shot_index, "path": path, "frame": int(local), "errors": []}
    try:
        shot = cd.d3d_shot_inputs(ROOT / path, index, with_labels=True)
        currents = shot.currents[local].astype(np.float64)
        ip_kA = float(shot.Ip_kA[local])
        sign = float(shot.ip_sign)
        tf_kA = float(shot.tf_kA[local])
        phi_gt = shot.phi[local].astype(np.float64)
        psi_gt = -phi_gt
        q95_gt, betaN_gt = (float(v) for v in shot.scalars[local])
        # ---- label reference (official extraction) ----------------------------------------------------- #
        C_gt = extract_lcfs(psi_gt, R, Z, "DIII-D", mask, maskf)
        if C_gt is None:
            info["errors"].append("label_no_lcfs")
            return rows, info
        d_gt = derive_frame(psi_gt, R, Z, "DIII-D", mask, maskf, contour=C_gt)
        rgeo = major_radius(C_gt)
        _, _, iz_gt, ir_gt = magnetic_axis(psi_gt, R, Z, "DIII-D", mask)
        xp_gt = limiting_xpoint(phi_gt, R, Z, iz_gt, ir_gt, mask)
        gt_ref = (C_gt, d_gt, rgeo, xp_gt)
        fit = cd3.fit_freegs_profile(phi_gt, solver, abs(ip_kA) * 1000.0)
        vac = solver.vacuum(sign * currents)
        n_idx, bz_axis = cd3.decay_index(vac, R, Z, d_gt["R_axis"], d_gt["Z_axis"])
        base = {k: v for k, v in cal.items() if k.startswith("cal_")}
        base |= {"shot_index": shot_index, "shot_key": shot.shot_key, "frame_index": int(shot.frames[local]), "time_ms": float(shot.times_ms[local]),
                "time_frac": float(local / max(shot.n_frames - 1, 1)), "ip_kA": ip_kA, "ip_sign": sign, "tf_kA": tf_kA,
                "q95_gt": q95_gt, "betaN_gt": betaN_gt, "ip_label_ratio": fit.ip_ratio, "fit_rel_rmse": fit.rel_rmse,
                "fit_rel_rmse_unweighted": fit.rel_rmse_unweighted, "fit_beta0": fit.params.beta0, "fit_alpha_p": fit.params.alpha_p,
                "fit_gamma_p": fit.params.gamma_p, "fit_alpha_f": fit.params.alpha_f, "fit_gamma_f": fit.params.gamma_f,
                "fit_beta_p": fit.beta_p_label, "fit_j_axis_ratio": fit.j_axis_ratio, "centroid_R_gt": fit.centroid_R, "centroid_Z_gt": fit.centroid_Z,
                "shafranov_shift_gt": d_gt["R_axis"] - fit.centroid_R, "decay_index_vac": n_idx, "Bz_vac_axis_T": bz_axis,
                "Zmax_gt": float(C_gt[:, 1].max()), "Zmin_gt": float(C_gt[:, 1].min()), "Rmin_gt": float(C_gt[:, 0].min()),
                "Rmax_gt": float(C_gt[:, 0].max()), "xpt_R_gt": xp_gt[0], "xpt_Z_gt": xp_gt[1], "xpt_n_gt": xp_gt[3],
                "psi_gt_sum": float(psi_gt.sum()), "psi_gt_sumsq": float(np.sum(psi_gt ** 2)), "psi_gt_n": int(psi_gt.size),
                "psi_gt_ptp": float(np.ptp(psi_gt))}
        for k in SCALARS:
            base[f"{k}_gt"] = d_gt[k]
        sampled = cs.ProfileParams.sample(rng, cd3.MAST_V1_RANGES)
        r_prior, z_prior = cd3.sample_centroid_prior(rng, setup.centre)
        base.update(r_prior=r_prior, z_prior=z_prior)
        results = {}

        def run(arm, params, r_t, z_t, mode, extra=None, slv=None, chord=None, beta_p_ref=np.nan, **solve_kw):
            slv = solver if slv is None else slv
            row = dict(base, arm=arm, status="ok", converged=False, seconds=np.nan, ecoil_shape_factor=slv.ecoil_shape_factor)
            t0 = time.monotonic()
            try:
                kw = dict(max_seconds=config["max_seconds"], extra_vacuum=extra, max_dI=config["max_dI"], max_iterations=config["max_iterations"])
                kw.update(solve_kw)
                if chord is None:
                    res = slv.solve_frame(currents, ip_kA, tf_kA, params, z_target=z_t, r_target=r_t, mode=mode, ip_sign=sign, **kw)
                else:
                    target_R, c_R, c_Z, match = chord
                    res, _ = cs.solve_frame_chord(slv, target_R, c_R, c_Z, (currents, ip_kA, tf_kA, params), match=match,
                                                  steps=config["chord_steps"], tol=config["chord_tol"], r_bounds=D3D_CHORD_R_BOUNDS,
                                                  r_target=r_t, z_target=z_t, mode=mode, ip_sign=sign, **kw)
                    row.update(r_seed=res["r_seed"], chord_R_target=res["chord_R_target"], chord_R_syn=res["chord_R_syn"], chord_steps=res["chord_steps"],
                               chord_match=match)
                    r_t = float(res["r_target"])          # the row's r_target is the FINAL pinned centroid target, r_seed the seed
                res["beta_p_ref"] = float(beta_p_ref)
                row["beta_p_ref"] = float(beta_p_ref)
                results[arm] = res
                row.update(converged=bool(res["converged"]), solve_status=res["status"], dI5_kAt=res["dI5_kAt"], dI6_kAt=res["dI6_kAt"],
                           beta0_final=res["params"].beta0, alpha_p=res["params"].alpha_p, gamma_p=res["params"].gamma_p,
                           alpha_f=res["params"].alpha_f, gamma_f=res["params"].gamma_f, r_target=r_t, z_target=z_t,
                           residual=res["residual"], iterations=res["iterations"], radial_free=bool(res["radial_free"]),
                           beta0_at_bound=bool(res.get("beta0_at_bound", False)), position_mode=res.get("position_mode", "pinned"),
                           beta_p_target=res.get("beta_p_target", np.nan), beta_p_achieved=res.get("beta_p_achieved", np.nan),
                           edge_amp=res["params"].edge_amp, edge_w=res["params"].edge_w)
                if res["surface"] is not None:
                    try:
                        scal = slv.scalars(res)
                        ok, reason = cs.accept_equilibrium(res, scal)
                        ok9, reason9 = cs.accept_equilibrium(res, scal, rules=dict(cs.ACCEPT, reject_beta0_bounds=True,
                                                                                    beta_p_consistency=config.get("beta_p_consistency", 0.0),
                                                                                    chord_tol=config.get("chord_accept_tol", 0.0)))
                        row.update(q95_pred=scal["q95"], betaN_pred=scal["betaN"], beta_p_pred=scal["beta_p"], accepted=bool(ok), accept_reason=reason,
                                   accepted_v9rule=bool(ok9), accept_reason_v9rule=reason9, li_pred_closure=scal["derived"]["li"],
                                   contact=scal["envelope_contact_fraction"], psi_ptp_pred=scal["psi_ptp"])
                        # current centroid of the solved equilibrium
                        row["centroid_R_pred"] = float(np.sum(res["j"] * slv.RR) / res["j"].sum())
                        row["centroid_Z_pred"] = float(np.sum(res["j"] * slv.ZZ) / res["j"].sum())
                        row.update({f"{k}_pred": v for k, v in cs.edge_current_ratio(res["j"], res["psi"], res["surface"]).items()})
                    except Exception as exc:  # noqa: BLE001
                        row.update(accepted=False, accept_reason=f"scalars_failed:{type(exc).__name__}")
                row.update(score_frame(psi_gt, res["psi"], R, Z, mask, maskf, gt_ref))
            except Exception as exc:  # noqa: BLE001
                row["status"] = f"exception:{type(exc).__name__}:{str(exc)[:80]}"
                info["errors"].append(f"{arm}:{traceback.format_exc()[-300:]}")
            row["seconds"] = time.monotonic() - t0
            rows.append(row)
            return row

        arms = config["arms"]
        if "sampled_centroid" in arms:
            run("sampled_centroid", sampled, r_prior, z_prior, "balance")
        if "sampled_efitpos" in arms:
            run("sampled_efitpos", sampled, fit.centroid_R, fit.centroid_Z, "balance")
        if "sampled_vertical" in arms:
            run("sampled_vertical", sampled, fit.centroid_R, fit.centroid_Z, "vertical")
        if "sampled_centroid_zlabel" in arms:
            run("sampled_centroid_zlabel", sampled, r_prior, fit.centroid_Z, "balance")
        if "sampledD3D_efitpos" in arms or "sampledD3D_centroid" in arms:
            sampled_d3d = cs.ProfileParams.sample(rng, D3D_FIT_RANGES)
            if "sampledD3D_efitpos" in arms:
                run("sampledD3D_efitpos", sampled_d3d, fit.centroid_R, fit.centroid_Z, "balance")
            if "sampledD3D_centroid" in arms:
                run("sampledD3D_centroid", sampled_d3d, r_prior, z_prior, "balance")
        # ---- v9 arms: position from force balance, beta_p from the frame's own Thomson data (label-free) ------------ #
        if any(a in arms for a in V9_ARMS):
            rng9 = np.random.default_rng(seed + 7)          # separate stream: the draws of the older arms are unchanged
            try:
                j_lab, surf_lab = solver.label_current(phi_gt)
                base.update({f"{k}_gt": v for k, v in cs.edge_current_ratio(j_lab, phi_gt, surf_lab).items()})
            except Exception as exc:  # noqa: BLE001
                info["errors"].append(f"label_edge:{type(exc).__name__}")
            proxy = None
            try:
                proxy = cs.thomson_beta_p_proxy(shot.chord_R, shot.chord_Z, shot.pe[local], shot.pe_mask[local], abs(ip_kA) * 1000.0,
                                                layout="d3d", d3d_a_est=config["d3d_a_est"])
            except ValueError as exc:
                info["errors"].append(f"proxy:{exc}")
            if proxy is not None:
                bp_target, bp_info = cs.sample_beta_p_target(proxy, rng9, tuple(config["pe_ratio_range"]), config["beta_p_jitter"])
                edge_amp, edge_w = cs.sample_edge_params(rng9, tuple(config["edge_amp_range"]), tuple(config["edge_w_range"]))
                base.update(beta_p_proxy_e=proxy["beta_p_e"], proxy_a_est=proxy["a_est"], proxy_R_axis_est=proxy["R_axis_est"],
                            proxy_p_axis_e=proxy["p_axis_e"], proxy_extent_ok=proxy["extent_ok"], beta_p_target_v9=bp_target,
                            beta_p_target_nojit=proxy["beta_p_e"] * bp_info["pe_ratio_proxy"], edge_amp_v9=edge_amp, edge_w_v9=edge_w, **bp_info)
                r_seed = float(config["r_init_d3d"])
                edge_params = cs.ProfileParams(*sampled.as_array(), edge_amp, edge_w)
                if "balance_v9" in arms:
                    run("balance_v9", sampled, r_seed, z_prior, "vertical", beta_p_target=bp_target, r_init=r_seed)
                if "balance_edge_v9" in arms:
                    run("balance_edge_v9", edge_params, r_seed, z_prior, "vertical", beta_p_target=bp_target, r_init=r_seed)
                if "balance_nojit_v9" in arms:
                    run("balance_nojit_v9", sampled, r_seed, z_prior, "vertical", beta_p_target=base["beta_p_target_nojit"], r_init=r_seed)
                if "balance_labelbp_v9" in arms:
                    run("balance_labelbp_v9", sampled, r_seed, z_prior, "vertical", beta_p_target=max(fit.beta_p_label, 0.03), r_init=r_seed)
                if "balance_edge_labelbp_v9" in arms:
                    run("balance_edge_labelbp_v9", edge_params, r_seed, z_prior, "vertical", beta_p_target=max(fit.beta_p_label, 0.03), r_init=r_seed)
                if "balance_edge_relax_v9" in arms:
                    run("balance_edge_relax_v9", edge_params, r_seed, z_prior, "vertical", beta_p_target=bp_target, r_init=r_seed, relaxation=0.4, max_iterations=160)
            # ---- iteration 3: chord-position arms (simulated midplane chord, see simulated_midplane_chord) -------------- #
            if any(a in arms for a in CHORD_ARMS):
                try:
                    c_R, c_Z, c_pe, c_m = simulated_midplane_chord(shot, local, phi_gt, solver, rng9, noise=config["midplane_noise"])
                    pm = cs.thomson_beta_p_proxy(c_R, c_Z, c_pe, c_m, abs(ip_kA) * 1000.0, layout="midplane")
                except ValueError as exc:
                    pm = None
                    info["errors"].append(f"midplane_chord:{exc}")
                if pm is not None:
                    bp_ref = pm["beta_p_e"] * 2.0          # mid-range total/electron ratio; the consistency band is wide
                    base.update(mid_R_axis_est=pm["R_axis_est"], mid_Rgeo_est=pm["Rgeo_est"], mid_a_est=pm["a_est"], mid_beta_p_e=pm["beta_p_e"],
                                mid_extent_ok=pm["extent_ok"], mid_R_in=pm["R_in"], mid_R_out=pm["R_out"])
                    if "edge_amp_v9" not in base:
                        edge_amp, edge_w = cs.sample_edge_params(rng9, tuple(config["edge_amp_range"]), tuple(config["edge_w_range"]))
                        base.update(edge_amp_v9=edge_amp, edge_w_v9=edge_w)
                    edge_params = cs.ProfileParams(*sampled.as_array(), base["edge_amp_v9"], base["edge_w_v9"])
                    off_axis, off_rgeo = config["chord_seed_offset_axis"], config["chord_seed_offset_rgeo"]

                    z_chord = fit.centroid_Z if config.get("chord_z", "label") == "label" else z_prior

                    def run_chord(arm, params, target, match, offset, noise_m=0.0):
                        tgt = float(target + (noise_m * rng9.standard_normal() if noise_m > 0 else 0.0))
                        seed = float(np.clip(tgt + offset, *D3D_CHORD_R_BOUNDS))
                        run(arm, params, seed, z_chord, "balance", chord=(tgt, c_R, c_Z, match), beta_p_ref=bp_ref)

                    if "chord_axis_v9" in arms:
                        run_chord("chord_axis_v9", sampled, pm["R_axis_est"], "axis", off_axis)
                    if "chord_rgeo_v9" in arms:
                        run_chord("chord_rgeo_v9", sampled, pm["Rgeo_est"], "rgeo", off_rgeo)
                    if "chord_axis_edge_v9" in arms:
                        run_chord("chord_axis_edge_v9", edge_params, pm["R_axis_est"], "axis", off_axis)
                    if "chord_rgeo_edge_v9" in arms:
                        run_chord("chord_rgeo_edge_v9", edge_params, pm["Rgeo_est"], "rgeo", off_rgeo)
                    if "chord_axis_noise_v9" in arms:
                        run_chord("chord_axis_noise_v9", sampled, pm["R_axis_est"], "axis", off_axis, noise_m=config["chord_noise"])
        if "fitted_efitpos" in arms:
            rowC = run("fitted_efitpos", fit.params, fit.centroid_R, fit.centroid_Z, "control")
        if "fitted_vertical" in arms:
            run("fitted_vertical", fit.params, fit.centroid_R, fit.centroid_Z, "vertical")
        if "fitted_rawcoil" in arms:
            run("fitted_rawcoil", fit.params, fit.centroid_R, fit.centroid_Z, "control", slv=raw)
        if ("fitted_bzfix" in arms or "fitted_multipole" in arms) and "fitted_efitpos" in results and results["fitted_efitpos"]["surface"] is not None:
            resC = results["fitted_efitpos"]
            label_surface = cs.plasma_surface(phi_gt, R, Z, mask, n_iter=22)
            mp_fit = cd3.exterior_multipole_fit(phi_gt, resC["psi"], solver, label_surface, resC["surface"])
            mp_row = {"mp_rms_raw": mp_fit["rms_raw"], "mp_rms_gauge": mp_fit["rms_after_1"], "mp_rms_bz": mp_fit["rms_after_2"],
                      "mp_rms_quad": mp_fit["rms_after_3"], "mp_rms_full": mp_fit["rms_after_5"], "mp_n": mp_fit["n_points"],
                      "mp_Bz_T_bzonly": mp_fit["uniform_Bz_T_bzonly"]} | {f"mp_{k}": v for k, v in mp_fit["coef"].items()}
            rowC.update(mp_row)
            if "fitted_bzfix" in arms:
                r = run("fitted_bzfix", fit.params, fit.centroid_R, fit.centroid_Z, "control", extra=mp_fit["correction_bz"])
                r.update(mp_row)
            if "fitted_multipole" in arms:
                r = run("fitted_multipole", fit.params, fit.centroid_R, fit.centroid_Z, "control", extra=mp_fit["correction_full"])
                r.update(mp_row)
        for arm, slv in (("oracle_j", solver), ("oracle_j_rawcoil", raw)):
            if arm not in arms:
                continue
            row = dict(base, arm=arm, status="ok", converged=True, seconds=np.nan, ecoil_shape_factor=slv.ecoil_shape_factor)
            t0 = time.monotonic()
            try:
                res = slv.oracle_solve(currents, ip_kA, phi_gt, ip_sign=sign)
                row.update(score_frame(psi_gt, res["psi"], R, Z, mask, maskf, gt_ref))
                row["dI5_kAt"] = row["dI6_kAt"] = 0.0
                res_raw = slv.oracle_solve(currents, ip_kA, phi_gt, ip_sign=sign, renormalise=False)
                sc = score_frame(psi_gt, res_raw["psi"], R, Z, mask, maskf, gt_ref)
                row["D_LCFS_labelIp"] = sc["D_LCFS"]
                row["r2_psi_frame_gauge_labelIp"] = sc["r2_psi_frame_gauge"]
            except Exception as exc:  # noqa: BLE001
                row["status"] = f"exception:{type(exc).__name__}:{str(exc)[:80]}"
                info["errors"].append(f"oracle:{traceback.format_exc()[-300:]}")
            row["seconds"] = time.monotonic() - t0
            rows.append(row)
    except Exception as exc:  # noqa: BLE001
        info["errors"].append(f"frame:{type(exc).__name__}:{str(exc)[:120]}:{traceback.format_exc()[-400:]}")
    info["seconds"] = time.monotonic() - started
    return rows, info


# ---------------------------------------------------------------------- #
def select_frames(data_dir: Path, n_shots: int, per_shot: int, min_ip: float, seed: int, index: cd.D3DFeatureIndex):
    manifest = json.loads((data_dir / "manifest.json").read_text())
    entries = manifest["train"]
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(entries))
    tasks = []
    chosen = 0
    for k in order:
        e = entries[k]
        with np.load(data_dir / e["path"], allow_pickle=False) as shot:
            X = shot["X"]
            ip = X[:, index.ip]
            ok = (np.abs(ip) > min_ip) & (X[:, index.ip_observed] > 0.5) & np.isfinite(shot["psi"]).all(axis=(1, 2))
        good = np.flatnonzero(ok)
        if len(good) < per_shot:
            continue
        picks = good[np.round(np.linspace(0, len(good) - 1, per_shot + 2)[1:-1]).astype(int)] if per_shot > 1 else good[[len(good) // 2]]
        for local in picks:
            tasks.append((str(Path(data_dir).resolve().relative_to(ROOT) / e["path"]), int(local),
                          int(rng.integers(0, 2 ** 31 - 1)), int(e["shot_index"])))
        chosen += 1
        if chosen >= n_shots:
            break
    return tasks


def q(values, keys=("median", "p10", "p90", "mean", "std", "count")):
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], float)
    if not len(v):
        return {"count": 0}
    out = {"count": int(len(v)), "median": float(np.median(v)), "p10": float(np.quantile(v, .1)), "p90": float(np.quantile(v, .9)),
           "mean": float(v.mean()), "std": float(v.std()), "min": float(v.min()), "max": float(v.max())}
    return out


def pooled_r2(gt, pred):
    gt, pred = np.asarray(gt, float), np.asarray(pred, float)
    m = np.isfinite(gt) & np.isfinite(pred)
    if m.sum() < 3:
        return None
    tot = float(np.sum((gt[m] - gt[m].mean()) ** 2))
    return float(1.0 - np.sum((gt[m] - pred[m]) ** 2) / tot) if tot > 0 else None


def corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 4 or a[m].std() == 0 or b[m].std() == 0:
        return None
    return float(np.corrcoef(a[m], b[m])[0, 1])


def summarise(rows, arms):
    by_arm = {a: [r for r in rows if r["arm"] == a] for a in arms}
    report = {"arms": {}}
    for a, rs in by_arm.items():
        n = len(rs)
        conv = [r for r in rs if r.get("converged") and r.get("status") == "ok"]
        scored = [r for r in conv if r.get("lcfs_found")]
        s = {"n_frames": n, "n_converged": len(conv), "n_lcfs_found": len(scored),
             "converged_fraction": len(conv) / max(n, 1),
             "accepted_fraction": float(np.mean([bool(r.get("accepted", False)) for r in conv])) if conv else None,
             "statuses": {}}
        for r in rs:
            key = r.get("status", "?") if r.get("status") != "ok" else ("converged" if r.get("converged") else f"not_converged:{r.get('solve_status')}")
            s["statuses"][key] = s["statuses"].get(key, 0) + 1
        if not scored:
            report["arms"][a] = s
            continue
        s["D_LCFS_all_converged"] = q([r["D_LCFS"] for r in conv])       # failures count as 1
        s["D_LCFS"] = q([r["D_LCFS"] for r in scored])
        s["hausdorff_m"] = q([r["D_LCFS"] * 0.5 * (r["Rmax_gt"] + r["Rmin_gt"]) for r in scored])
        for k in SCALARS:
            s[f"{k}_err"] = q([r[f"{k}_err"] for r in scored])
            s[f"{k}_abs_err"] = q([abs(r[f"{k}_err"]) for r in scored])
            s[f"{k}_r2_pooled"] = pooled_r2([r[f"{k}_gt"] for r in scored], [r[f"{k}_pred"] for r in scored])
        for k in ("Zmax_err", "Zmin_err", "Rmin_err", "Rmax_err", "xpt_R_err", "xpt_Z_err", "xpt_dist", "ptp_ratio", "psi_gauge_offset",
                  "psi_rms_gauge", "r2_psi_frame_raw", "r2_psi_frame_gauge", "dI5_kAt", "dI6_kAt", "beta0_final", "seconds", "iterations", "residual",
                  "beta_p_target", "beta_p_achieved", "beta_p_pred", "j_edge_over_axis_pred", "edge_I_frac_s_gt_0.9_pred", "li_pred", "edge_amp",
                  "chord_steps", "chord_R_syn", "chord_R_target", "r_seed"):
            if any(k in r for r in scored):
                s[k] = q([r.get(k, np.nan) for r in scored])
        if any("chord_R_target" in r for r in scored):
            s["chord_target_minus_syn"] = q([r["chord_R_target"] - r["chord_R_syn"] for r in scored if "chord_R_target" in r])
            s["chord_target_minus_syn_abs"] = q([abs(r["chord_R_target"] - r["chord_R_syn"]) for r in scored if "chord_R_target" in r])
            s["r_target_minus_seed"] = q([r["r_target"] - r["r_seed"] for r in scored if "r_seed" in r])
            s["beta_p_pred_over_ref"] = q([r["beta_p_pred"] / r["beta_p_ref"] for r in scored if r.get("beta_p_ref", 0) > 0 and "beta_p_pred" in r])
        # v9 bookkeeping: bound hits, v9 acceptance rule, beta_p target vs label, edge current vs label
        if any("beta0_at_bound" in r for r in conv):
            acc9 = [r for r in scored if r.get("accepted_v9rule")]
            if acc9:
                s["D_LCFS_accepted_v9rule"] = q([r["D_LCFS"] for r in acc9])
                s["centroid_R_err_accepted_v9rule"] = q([r["centroid_R_pred"] - r["centroid_R_gt"] for r in acc9 if "centroid_R_pred" in r])
                s["kappa_err_accepted_v9rule"] = q([r["kappa_err"] for r in acc9])
            bal = [r for r in scored if r.get("radial_free")]
            if bal:
                s["D_LCFS_balanced"] = q([r["D_LCFS"] for r in bal])
                s["centroid_R_err_balanced"] = q([r["centroid_R_pred"] - r["centroid_R_gt"] for r in bal if "centroid_R_pred" in r])
            s["balanced_fraction"] = float(np.mean([bool(r.get("radial_free", False)) for r in conv]))
            s["beta0_at_bound_fraction"] = float(np.mean([bool(r.get("beta0_at_bound", False)) for r in conv]))
            s["accepted_v9rule_fraction"] = float(np.mean([bool(r.get("accepted_v9rule", False)) for r in conv]))
            reasons = {}
            for r in conv:
                reasons[r.get("accept_reason_v9rule", "?")] = reasons.get(r.get("accept_reason_v9rule", "?"), 0) + 1
            s["accept_reasons_v9rule"] = reasons
        if any(np.isfinite(r.get("beta_p_target", np.nan)) for r in scored):
            s["beta_p_target_over_fit_beta_p"] = q([r["beta_p_target"] / r["fit_beta_p"] for r in scored if np.isfinite(r.get("beta_p_target", np.nan)) and r["fit_beta_p"] > 0])
            s["log_beta_p_target_over_fit"] = q([np.log(r["beta_p_target"] / r["fit_beta_p"]) for r in scored if np.isfinite(r.get("beta_p_target", np.nan)) and r["fit_beta_p"] > 0])
            s["beta_p_match_rel"] = q([(r["beta_p_achieved"] - r["beta_p_target"]) / r["beta_p_target"] for r in scored if np.isfinite(r.get("beta_p_achieved", np.nan))])
        if any("j_edge_over_axis_pred" in r for r in scored):
            s["j_edge_over_axis_err"] = q([r["j_edge_over_axis_pred"] - r.get("j_edge_over_axis_gt", np.nan) for r in scored])
        for k in ("dI5_kAt", "dI6_kAt"):
            if any(k in r for r in scored):
                s[f"abs_{k}"] = q([abs(r.get(k, np.nan)) for r in scored])
        if any("q95_pred" in r for r in scored):
            s["q95_r2_pooled"] = pooled_r2([r["q95_gt"] for r in scored], [r.get("q95_pred", np.nan) for r in scored])
            s["betaN_r2_pooled"] = pooled_r2([r["betaN_gt"] for r in scored], [r.get("betaN_pred", np.nan) for r in scored])
            s["q95_rel_err"] = q([(r.get("q95_pred", np.nan) - r["q95_gt"]) / r["q95_gt"] for r in scored])
            s["betaN_rel_err"] = q([(r.get("betaN_pred", np.nan) - r["betaN_gt"]) / r["betaN_gt"] for r in scored])
        if any("centroid_R_pred" in r for r in scored):
            s["centroid_R_err"] = q([r["centroid_R_pred"] - r["centroid_R_gt"] for r in scored if "centroid_R_pred" in r])
            s["centroid_R_abs_err"] = q([abs(r["centroid_R_pred"] - r["centroid_R_gt"]) for r in scored if "centroid_R_pred" in r])
            s["centroid_Z_err"] = q([r["centroid_Z_pred"] - r["centroid_Z_gt"] for r in scored if "centroid_R_pred" in r])
        # pooled psi R2 (scorer-style pooled mean over the scored frames of this arm)
        gsum = sum(r["psi_gt_sum"] for r in scored)
        gsq = sum(r["psi_gt_sumsq"] for r in scored)
        gn = sum(r["psi_gt_n"] for r in scored)
        tot = gsq - gsum ** 2 / gn
        s["r2_psi_pooled_raw"] = float(1 - sum(r["psi_ss_res_raw"] for r in scored) / tot)
        s["r2_psi_pooled_gauge"] = float(1 - sum(r["psi_ss_res_gauge"] for r in scored) / tot)
        # dependence of the shape error on control effort / decay index / fit quality
        dep = {}
        for x in ("dI5_kAt", "dI6_kAt"):
            if any(x in r for r in scored):
                dep[f"corr_D_LCFS_abs_{x}"] = corr([abs(r.get(x, np.nan)) for r in scored], [r["D_LCFS"] for r in scored])
                dep[f"corr_kappa_err_{x}"] = corr([r.get(x, np.nan) for r in scored], [r["kappa_err"] for r in scored])
                dep[f"corr_Zmax_err_{x}"] = corr([r.get(x, np.nan) for r in scored], [r["Zmax_err"] for r in scored])
                dep[f"corr_Zmin_err_{x}"] = corr([r.get(x, np.nan) for r in scored], [r["Zmin_err"] for r in scored])
        dep["corr_D_LCFS_decay_index"] = corr([r["decay_index_vac"] for r in scored], [r["D_LCFS"] for r in scored])
        dep["corr_kappa_err_decay_index"] = corr([r["decay_index_vac"] for r in scored], [r["kappa_err"] for r in scored])
        dep["corr_D_LCFS_fit_rel_rmse"] = corr([r["fit_rel_rmse"] for r in scored], [r["D_LCFS"] for r in scored])
        dep["corr_D_LCFS_kappa_gt"] = corr([r["kappa_gt"] for r in scored], [r["D_LCFS"] for r in scored])
        dep["corr_D_LCFS_ip_kA"] = corr([abs(r["ip_kA"]) for r in scored], [r["D_LCFS"] for r in scored])
        dep["corr_D_LCFS_time_frac"] = corr([r["time_frac"] for r in scored], [r["D_LCFS"] for r in scored])
        dep["corr_D_LCFS_ip_label_ratio"] = corr([r["ip_label_ratio"] for r in scored], [r["D_LCFS"] for r in scored])
        # decay-index terciles
        n_idx = np.asarray([r["decay_index_vac"] for r in scored])
        dl = np.asarray([r["D_LCFS"] for r in scored])
        ke = np.asarray([r["kappa_err"] for r in scored])
        fin = np.isfinite(n_idx)
        if fin.sum() > 9:
            edges = np.quantile(n_idx[fin], [0, 1 / 3, 2 / 3, 1])
            dep["decay_index_terciles"] = []
            for lo, hi in zip(edges[:-1], edges[1:]):
                m = fin & (n_idx >= lo) & (n_idx <= hi)
                dep["decay_index_terciles"].append({"n_range": [float(lo), float(hi)], "count": int(m.sum()), "D_LCFS_median": float(np.median(dl[m])),
                                                    "kappa_err_mean": float(np.mean(ke[m]))})
        s["dependence"] = dep
        if any("mp_rms_raw" in r for r in scored):
            mp = [r for r in scored if "mp_rms_raw" in r]
            s["exterior_multipole"] = {k: q([r[k] for r in mp]) for k in ("mp_rms_raw", "mp_rms_gauge", "mp_rms_bz", "mp_rms_quad", "mp_rms_full",
                                                                        "mp_Bz_T_bzonly", "mp_uniform_Bz_T", "mp_quadrupole", "mp_odd_Z", "mp_odd_R2Z", "mp_gauge")}
            s["exterior_multipole"]["fraction_rms_removed_by_gauge"] = q([1 - r["mp_rms_gauge"] / r["mp_rms_raw"] for r in mp])
            s["exterior_multipole"]["fraction_rms_removed_by_uniform_Bz_after_gauge"] = q([1 - r["mp_rms_bz"] / r["mp_rms_gauge"] for r in mp])
            s["exterior_multipole"]["fraction_rms_removed_by_quadrupole_after_Bz"] = q([1 - r["mp_rms_quad"] / r["mp_rms_bz"] for r in mp])
            s["exterior_multipole"]["fraction_rms_removed_by_odd_after_quad"] = q([1 - r["mp_rms_full"] / r["mp_rms_quad"] for r in mp])
        if a == "oracle_j" and any("D_LCFS_labelIp" in r for r in scored):
            s["D_LCFS_labelIp"] = q([r["D_LCFS_labelIp"] for r in scored if "D_LCFS_labelIp" in r])
            s["r2_psi_frame_gauge_labelIp"] = q([r["r2_psi_frame_gauge_labelIp"] for r in scored if "r2_psi_frame_gauge_labelIp" in r])
        report["arms"][a] = s
    # frame-level label facts
    first = {}
    for r in rows:      # one merged dict per frame (label facts are shared; v9 proxy facts only appear on the v9 rows)
        key = (r["shot_index"], r["frame_index"])
        first[key] = dict(r, **first[key]) if key in first else dict(r)
    fr = list(first.values())
    report["labels"] = {"n_frames": len(fr), "n_shots": len(set(r["shot_index"] for r in fr)),
                        "ip_kA_abs": q([abs(r["ip_kA"]) for r in fr]), "negative_ip_fraction": float(np.mean([r["ip_sign"] < 0 for r in fr])),
                        "kappa_gt": q([r["kappa_gt"] for r in fr]), "li_gt": q([r["li_gt"] for r in fr]), "R_axis_gt": q([r["R_axis_gt"] for r in fr]),
                        "Z_axis_gt": q([r["Z_axis_gt"] for r in fr]), "centroid_R_gt": q([r["centroid_R_gt"] for r in fr]),
                        "centroid_Z_gt": q([r["centroid_Z_gt"] for r in fr]), "shafranov_shift_gt": q([r["shafranov_shift_gt"] for r in fr]),
                        "q95_gt": q([r["q95_gt"] for r in fr]), "betaN_gt": q([r["betaN_gt"] for r in fr]),
                        "ip_label_over_measured": q([r["ip_label_ratio"] for r in fr]), "decay_index_vac_at_axis": q([r["decay_index_vac"] for r in fr]),
                        "Bz_vac_axis_T": q([r["Bz_vac_axis_T"] for r in fr]), "xpt_found_fraction": float(np.mean([r["xpt_n_gt"] > 0 for r in fr])),
                        "xpt_Z_gt": q([r["xpt_Z_gt"] for r in fr]), "psi_gt_ptp": q([r["psi_gt_ptp"] for r in fr]),
                        "profile_fit": {"rel_rmse_weighted": q([r["fit_rel_rmse"] for r in fr]), "rel_rmse_unweighted": q([r["fit_rel_rmse_unweighted"] for r in fr]),
                                        "beta0": q([r["fit_beta0"] for r in fr]), "alpha_p": q([r["fit_alpha_p"] for r in fr]), "gamma_p": q([r["fit_gamma_p"] for r in fr]),
                                        "alpha_f": q([r["fit_alpha_f"] for r in fr]), "gamma_f": q([r["fit_gamma_f"] for r in fr]), "beta_p": q([r["fit_beta_p"] for r in fr]),
                                        "j_axis_ratio": q([r["fit_j_axis_ratio"] for r in fr]),
                                        "fraction_inside_mast_v1_ranges": {k: float(np.mean([lo <= r[f"fit_{k}"] <= hi for r in fr]))
                                                                           for k, (lo, hi) in cd3.MAST_V1_RANGES.items()}},
                        "position_prior": {"r_prior_minus_centroid_gt": q([r["r_prior"] - r["centroid_R_gt"] for r in fr]),
                                           "z_prior_minus_centroid_gt": q([r["z_prior"] - r["centroid_Z_gt"] for r in fr])}}
    frm = [r for r in fr if "mid_R_axis_est" in r]
    if frm:
        report["labels"]["simulated_midplane_chord"] = {
            "n_frames": len(frm), "note": "DIII-D Thomson p_e mapped to label flux and re-sampled on a MAST-like Z = 0 chord (validation-only)",
            "R_axis_est_minus_R_axis_gt": q([r["mid_R_axis_est"] - r["R_axis_gt"] for r in frm]),
            "Rgeo_est_minus_Rgeo_gt": q([r["mid_Rgeo_est"] - 0.5 * (r["Rmax_gt"] + r["Rmin_gt"]) for r in frm]),
            "R_out_minus_Rmax_gt": q([r["mid_R_out"] - r["Rmax_gt"] for r in frm]), "R_in_minus_Rmin_gt": q([r["mid_R_in"] - r["Rmin_gt"] for r in frm]),
            "a_est_minus_a_gt": q([r["mid_a_est"] - 0.5 * (r["Rmax_gt"] - r["Rmin_gt"]) for r in frm]),
            "R_axis_est_minus_centroid_gt": q([r["mid_R_axis_est"] - r["centroid_R_gt"] for r in frm]),
            "Rgeo_est_minus_centroid_gt": q([r["mid_Rgeo_est"] - r["centroid_R_gt"] for r in frm]),
            "beta_p_e_x2_over_fit_beta_p": q([2 * r["mid_beta_p_e"] / r["fit_beta_p"] for r in frm if r["fit_beta_p"] > 0]),
            "corr_log_mid_proxy_log_fit": corr([np.log(max(r["mid_beta_p_e"], 1e-6)) for r in frm], [np.log(max(r["fit_beta_p"], 1e-6)) for r in frm]),
            "extent_ok_fraction": float(np.mean([bool(r["mid_extent_ok"]) for r in frm]))}
    fr9 = [r for r in fr if "beta_p_proxy_e" in r]
    if fr9:
        report["labels"]["thomson_beta_p_proxy"] = {
            "n_frames_with_proxy": len(fr9), "n_frames_without_proxy": len(fr) - len(fr9),
            "beta_p_e_raw": q([r["beta_p_proxy_e"] for r in fr9]), "pe_ratio_sampled": q([r["pe_ratio_proxy"] for r in fr9]),
            "jitter": q([r["beta_p_jitter"] for r in fr9]), "target": q([r["beta_p_target_v9"] for r in fr9]),
            "target_over_fit_beta_p": q([r["beta_p_target_v9"] / r["fit_beta_p"] for r in fr9 if r["fit_beta_p"] > 0]),
            "nojit_over_fit_beta_p": q([r["beta_p_target_nojit"] / r["fit_beta_p"] for r in fr9 if r["fit_beta_p"] > 0]),
            "proxy_x2_over_fit_beta_p": q([2 * r["beta_p_proxy_e"] / r["fit_beta_p"] for r in fr9 if r["fit_beta_p"] > 0]),
            "corr_log_proxy_log_fit": corr([np.log(max(r["beta_p_proxy_e"], 1e-6)) for r in fr9], [np.log(max(r["fit_beta_p"], 1e-6)) for r in fr9]),
            "proxy_R_axis_est_minus_R_axis_gt": q([r["proxy_R_axis_est"] - r["R_axis_gt"] for r in fr9]),
            "proxy_a_est_minus_a_gt": q([r["proxy_a_est"] - 0.5 * (r["Rmax_gt"] - r["Rmin_gt"]) for r in fr9]),
            "edge_amp_sampled": q([r["edge_amp_v9"] for r in fr9]), "edge_w_sampled": q([r["edge_w_v9"] for r in fr9]),
            "label_j_edge_over_axis": q([r["j_edge_over_axis_gt"] for r in fr9 if "j_edge_over_axis_gt" in r]),
            "label_edge_I_frac_s_gt_0.9": q([r["edge_I_frac_s_gt_0.9_gt"] for r in fr9 if "edge_I_frac_s_gt_0.9_gt" in r])}
    # v9 comparison table (one line per arm)
    table = {}
    for a, s in report["arms"].items():
        if "D_LCFS" in s:
            table[a] = {"n_conv": s["n_converged"], "conv_frac": round(s["converged_fraction"], 3), "accepted_frac": s.get("accepted_fraction"),
                        "accepted_v9rule_frac": s.get("accepted_v9rule_fraction"), "D_LCFS_median": round(s["D_LCFS"]["median"], 4),
                        "D_LCFS_p90": round(s["D_LCFS"]["p90"], 4), "D_LCFS_mean": round(s["D_LCFS"]["mean"], 4),
                        "kappa_err_mean": round(s["kappa_err"]["mean"], 4), "kappa_err_median": round(s["kappa_err"]["median"], 4),
                        "centroid_R_err_median": round(s["centroid_R_err"]["median"], 4) if "centroid_R_err" in s else None,
                        "centroid_R_err_p10_p90": [round(s["centroid_R_err"]["p10"], 4), round(s["centroid_R_err"]["p90"], 4)] if "centroid_R_err" in s else None,
                        "li_err_mean": round(s["li_err"]["mean"], 4), "R_axis_err_median": round(s["R_axis_err"]["median"], 4),
                        "beta0_at_bound_frac": s.get("beta0_at_bound_fraction"), "balanced_frac": s.get("balanced_fraction"),
                        "D_LCFS_balanced_median_p90": [round(s["D_LCFS_balanced"]["median"], 4), round(s["D_LCFS_balanced"]["p90"], 4)] if "D_LCFS_balanced" in s else None,
                        "D_LCFS_accepted_v9_median_p90_n": [round(s["D_LCFS_accepted_v9rule"]["median"], 4), round(s["D_LCFS_accepted_v9rule"]["p90"], 4), s["D_LCFS_accepted_v9rule"]["count"]] if "D_LCFS_accepted_v9rule" in s else None,
                        "centroid_R_err_balanced_median": round(s["centroid_R_err_balanced"]["median"], 4) if "centroid_R_err_balanced" in s else None,
                        "j_edge_over_axis_pred_median_p90": [round(s["j_edge_over_axis_pred"]["median"], 4), round(s["j_edge_over_axis_pred"]["p90"], 4)] if "j_edge_over_axis_pred" in s else None,
                        "centroid_R_abs_err_median": round(s["centroid_R_abs_err"]["median"], 4) if "centroid_R_abs_err" in s else None,
                        "chord_target_minus_syn_abs_median": round(s["chord_target_minus_syn_abs"]["median"], 4) if "chord_target_minus_syn_abs" in s else None,
                        "accept_reasons_v9rule": s.get("accept_reasons_v9rule"),
                        "r2_psi_gauge": round(s["r2_psi_pooled_gauge"], 4)}
    report["comparison_table"] = table
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", type=Path, default=ROOT / "results/analysis/synth_fidelity_d3d")
    p.add_argument("--data-dir", type=Path, default=ROOT / "data/advanced_v1")
    p.add_argument("--shots", type=int, default=100)
    p.add_argument("--frames-per-shot", type=int, default=3)
    p.add_argument("--min-ip-kA", type=float, default=300.0)
    p.add_argument("--seed", type=int, default=20260925)
    p.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    p.add_argument("--max-seconds", type=float, default=120.0)
    p.add_argument("--lcfs-iter", type=int, default=14)
    p.add_argument("--arms", type=str, default=",".join(ARMS + EXTRA_ARMS))
    p.add_argument("--tag", type=str, default="")
    p.add_argument("--ecoil-shape-factor", type=str, default="auto", help="'auto' = median exterior-regression factor of the calibration pass, or a number")
    p.add_argument("--max-dI", type=float, default=150.0)
    p.add_argument("--max-iterations", type=int, default=120)
    # ---- v9 arm options (label-free beta_p proxy, position from balance, edge current) ----------------------------- #
    p.add_argument("--pe-ratio-range", type=float, nargs=2, default=(1.5, 2.5), help="total/electron pressure ratio of the beta_p proxy, U(range)")
    p.add_argument("--beta-p-jitter", type=float, default=0.3, help="lognormal sigma applied to the proxy beta_p target (balance_v9 arms)")
    p.add_argument("--r-init", type=float, default=1.65, help="label-free seed radius of the initial current in the balance arms (m)")
    p.add_argument("--d3d-a-est", type=float, default=0.58, help="minor-radius prior of the DIII-D proxy layout (m)")
    p.add_argument("--edge-amp-range", type=float, nargs=2, default=(0.0, 0.3))
    p.add_argument("--edge-w-range", type=float, nargs=2, default=(0.03, 0.12))
    # ---- iteration 3: chord-position arms ------------------------------------------------------------------------- #
    p.add_argument("--chord-steps", type=int, default=2, help="max warm-started position-correction re-solves of the chord arms")
    p.add_argument("--chord-tol", type=float, default=0.005, help="stop the chord correction when |target - observable| < tol (m)")
    p.add_argument("--chord-accept-tol", type=float, default=0.02, help="v9 acceptance: |target - observable| must be <= this (m); 0 = off")
    p.add_argument("--chord-noise", type=float, default=0.02, help="sigma (m) of the target perturbation of chord_axis_noise_v9")
    p.add_argument("--chord-seed-offset-axis", type=float, default=-0.05, help="seed centroid = R_axis_est + this (m); DIII-D R_axis - centroid median 0.053")
    p.add_argument("--chord-seed-offset-rgeo", type=float, default=0.0, help="seed centroid = Rgeo_est + this (m)")
    p.add_argument("--midplane-noise", type=float, default=0.05, help="multiplicative noise of the simulated midplane chord")
    p.add_argument("--beta-p-consistency", type=float, default=2.5, help="v9 acceptance: balanced beta_p within [1/f, f] x (2 x chord beta_p,e); 0 = off")
    p.add_argument("--chord-z", choices=("label", "prior"), default="label", help="vertical centroid target of the chord arms (see the note at CHORD_ARMS)")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    arms = [a for a in args.arms.split(",") if a]
    config = {"data_dir": str(Path(args.data_dir).resolve().relative_to(ROOT)), "lcfs_iter": args.lcfs_iter, "max_seconds": args.max_seconds,
              "arms": arms, "max_dI": args.max_dI, "max_iterations": args.max_iterations, "ecoil_shape_factor": 1.0,
              "pe_ratio_range": tuple(args.pe_ratio_range), "beta_p_jitter": args.beta_p_jitter, "r_init_d3d": args.r_init,
              "d3d_a_est": args.d3d_a_est, "edge_amp_range": tuple(args.edge_amp_range), "edge_w_range": tuple(args.edge_w_range),
              "chord_steps": args.chord_steps, "chord_tol": args.chord_tol, "chord_accept_tol": args.chord_accept_tol, "chord_noise": args.chord_noise,
              "chord_seed_offset_axis": args.chord_seed_offset_axis, "chord_seed_offset_rgeo": args.chord_seed_offset_rgeo,
              "midplane_noise": args.midplane_noise, "beta_p_consistency": args.beta_p_consistency, "chord_z": args.chord_z}
    setup = cd.build_machine_setup("DIII-D")
    manifest = json.loads((args.data_dir / "manifest.json").read_text())
    index = cd.D3DFeatureIndex(manifest["feature_names"], setup.table_names)
    tasks = select_frames(args.data_dir, args.shots, args.frames_per_shot, args.min_ip_kA, args.seed, index)
    print(json.dumps({"frames": len(tasks), "shots": len(set(t[3] for t in tasks)), "arms": arms, "workers": args.workers,
                      "z_ctrl": cd3.Z_CTRL, "r_ctrl": cd3.R_CTRL, "R0": cd3.R0_D3D, "ranges": cd3.MAST_V1_RANGES}), flush=True)
    started = time.monotonic()
    ctx = mp.get_context("fork")
    # ---- calibration pass: linear coil-model check on every frame (raw tables) ---------------------------------- #
    with ctx.Pool(args.workers, initializer=_init, initargs=(config,)) as pool:
        cals = list(pool.imap_unordered(calibrate_frame, tasks, chunksize=2))
    good = [c for c in cals if "cal_f_E" in c]
    calibration = {"n_frames": len(good), "n_failed": len(cals) - len(good),
                   "f_F_exterior": q([c["cal_f_F"] for c in good]), "f_E_exterior": q([c["cal_f_E"] for c in good]),
                   "f_E_axis_budget": q([c["cal_f_E_axis"] for c in good]), "gauge_Wb_per_rad": q([c["cal_gauge"] for c in good]),
                   "exterior_rms_raw_model": q([c["cal_rms_raw"] for c in good]), "exterior_rms_after_fit": q([c["cal_rms_fit"] for c in good]),
                   "Bz_axis_G": {k: q([c[f"cal_Bz_{k}_G"] for c in good]) for k in ("F", "E", "plasma", "label", "residual_raw")},
                   "corr_f_E_vs_ecoil_kA": corr([c["ecoil_kA"] for c in good], [c["cal_f_E"] for c in good]),
                   "corr_f_E_vs_ip_kA": corr([abs(c["ip_kA"]) for c in good], [c["cal_f_E"] for c in good])}
    if args.ecoil_shape_factor == "auto":
        factor = float(np.median([c["cal_f_E"] for c in good])) if good else 1.0
    else:
        factor = float(args.ecoil_shape_factor)
    calibration["ecoil_shape_factor_used"] = factor
    print(json.dumps({"calibration": {k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if kk in ("median", "p10", "p90", "count")})
                                      for k, v in calibration.items() if k != "Bz_axis_G"}, "ecoil_shape_factor_used": factor,
                      "calibration_seconds": time.monotonic() - started}), flush=True)
    config = dict(config, ecoil_shape_factor=factor, calibration={f"{c['shot_index']}_{int(c['frame_index'])}": c for c in good})
    rows, infos = [], []
    with ctx.Pool(args.workers, initializer=_init, initargs=(config,)) as pool:
        for done, (rs, info) in enumerate(pool.imap_unordered(process_frame, tasks, chunksize=1)):
            rows.extend(rs)
            infos.append(info)
            if (done + 1) % 10 == 0 or done + 1 == len(tasks):
                el = time.monotonic() - started
                conv = sum(1 for r in rows if r.get("converged"))
                print(f"{done + 1}/{len(tasks)} frames, {len(rows)} rows, {conv} converged, {el:.0f} s", flush=True)
    elapsed = time.monotonic() - started
    keys = sorted({k for r in rows for k in r})
    tag = f"_{args.tag}" if args.tag else ""
    with (args.output / f"frames{tag}.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if (isinstance(v, float) and not np.isfinite(v)) else v) for k, v in r.items()})
    report = summarise(rows, arms)
    report["arm_definitions"] = ARM_DEFINITIONS
    report["coil_model_calibration"] = calibration
    report["config"] = {"created_utc": datetime.now(timezone.utc).isoformat(), "arguments": {k: str(v) for k, v in vars(args).items()},
                        "elapsed_seconds": elapsed, "frames_requested": len(tasks), "z_control_pair": cd3.Z_CTRL, "r_control_pair": cd3.R_CTRL,
                        "R0_profile_m": cd3.R0_D3D, "mast_v1_ranges": cd3.MAST_V1_RANGES, "position_prior": "MAST v1 r_target U(0.76,0.90)/0.8397 and"
                        " z N(0,0.02)/0.8397 clipped 0.08/0.8397, scaled to the DIII-D envelope centroid (1.590, -0.049)",
                        "label_informed_position": "current centroid of j_EFIT = -Delta* phi_EFIT/(mu0 R) inside the official LCFS (R_axis - centroid = Shafranov shift, reported)",
                        "scorer": "starter/fusion_scoring (extract_lcfs / derive_frame / symmetric_hausdorff), DIII-D convention, prediction psi = -phi, axis_sign -1;"
                                  " D_LCFS normalised by the per-frame label major radius",
                        "reference_C1_dev": {"D_LCFS_mean": 0.0077, "D_LCFS_median": 0.0042, "D_LCFS_p90": 0.0209, "kappa_r2": 0.980, "li_r2": 0.970,
                                             "R_axis_r2": 0.983, "volume_r2": 0.991, "source": "results/analysis/c1_error_anatomy_083/report.json baseline"},
                        "reference_MAST_official": {"D_LCFS": 0.2247, "consistency": 0.141, "r2_psi": 0.870, "source": "RESUME.md dual 20260925-078"},
                        "reference_synthetic_MAST_heldout": {"D_LCFS": 0.021, "consistency": 0.959}}
    report["frame_errors"] = [i for i in infos if i["errors"]][:60]
    report["n_frame_errors"] = sum(1 for i in infos if i["errors"])
    (args.output / f"report{tag}.json").write_text(json.dumps(report, indent=1, default=lambda o: None if (isinstance(o, float) and not np.isfinite(o)) else (o.item() if hasattr(o, "item") else str(o))) + "\n")
    for a in arms:
        s = report["arms"].get(a, {})
        if "D_LCFS" in s:
            print(a, f"conv {s['converged_fraction']:.2f}", f"D_LCFS med {s['D_LCFS']['median']:.4f} p90 {s['D_LCFS']['p90']:.4f}",
                  f"kappa_err mean {s['kappa_err']['mean']:+.3f} std {s['kappa_err']['std']:.3f}", f"kappa R2 {s['kappa_r2_pooled']}",
                  f"r2psi gauge {s['r2_psi_pooled_gauge']:.4f} raw {s['r2_psi_pooled_raw']:.4f}", flush=True)
        else:
            print(a, s.get("statuses"), flush=True)
    print(f"elapsed {elapsed:.0f} s; wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
