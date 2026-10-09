"""Shared helpers for the q95 physics-scaling fix (fit_q95_scaling.py / apply_q95_scaling.py).

Boundary geometry comes from the official scorer's LCFS extraction (starter/fusion_scoring/lcfs.py)
on a flux map; B_T from the TF coil current (kA, published turn counts); Ip from the magnetics.
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "starter" / "fusion_scoring"))
from lcfs import extract_lcfs  # noqa: E402

MU0 = 4e-7 * np.pi
TF_TURNS = {"DIII-D": 144, "MAST": 24}          # fusion/profile_scalars.py (published hardware)
GEOM_NAMES = ["a", "R0", "kappa", "tri_top", "tri_bot"]
# Fitted models. The DIII-D training data span eps = a/R0 of only 0.30-0.42, so the eps dependence is NOT
# identifiable from DIII-D and any free eps coefficient extrapolates arbitrarily to MAST (eps ~ 0.77): the
# free-eps models (loglin6, loglin_fixed) are kept for reference only. The "pinned" models fix the
# aspect-ratio factor to a physics form f(eps) and fit only c0, c2 (log kappa) and c5 (delta) on DIII-D:
#     log q95 = log q_cyl + log f(eps) + c0 + c2 log kappa + c5 delta
#     pinned_p050 / pinned_p075 / pinned_p100 : f = (1 - eps^2)^-p, p = 0.5 / 0.75 / 1.0
#     wesson                                  : f = 1 + eps^2        (Wesson toroidal correction, Lambda = 0)
#     iter89_shape                            : f = (1.17 - 0.65 eps) / (1 - eps^2)^2  (ITER-89 / Uckan)
# (1 - eps^2)^-p reduces to 1 + p eps^2 at small eps (Wesson's leading order) and at eps ~ 0.77 gives the
# q95/q_cyl ~ 2 that spherical tokamaks (MAST/NSTX, q95 ~ 5-7 at q_cyl ~ 3) exhibit for p ~ 0.75; the ITER-89
# (1 - eps^2)^-2 form was derived for conventional aspect ratio and gives q95/q_cyl ~ 4 at eps 0.77.
MODEL_NAMES = ["loglin6", "loglin_fixed", "iter89", "iter89_fixed",
               "pinned_p050", "pinned_p075", "pinned_p100", "wesson", "iter89_shape"]
MODEL_FORMULAS = {
    "loglin6": "log q95 = c0 + c1 log q_cyl + c2 log kappa + c3 eps + c4 eps^2 + c5 delta",
    "loglin_fixed": "log q95 = log q_cyl + c0 + c2 log kappa + c3 eps + c5 delta",
    "iter89": "log q95 = c0 + c1 log q_iter89",
    "iter89_fixed": "log q95 = c0 + log q_iter89",
    "pinned_p050": "log q95 = log q_cyl - 0.5 log(1 - eps^2) + c0 + c2 log kappa + c5 delta",
    "pinned_p075": "log q95 = log q_cyl - 0.75 log(1 - eps^2) + c0 + c2 log kappa + c5 delta",
    "pinned_p100": "log q95 = log q_cyl - 1.0 log(1 - eps^2) + c0 + c2 log kappa + c5 delta",
    "wesson": "log q95 = log q_cyl + log(1 + eps^2) + c0 + c2 log kappa + c5 delta",
    "iter89_shape": "log q95 = log q_cyl + log((1.17 - 0.65 eps) / (1 - eps^2)^2) + c0 + c2 log kappa + c5 delta",
}
PINNED_EXPONENT = {"pinned_p050": 0.5, "pinned_p075": 0.75, "pinned_p100": 1.0}
Q95_CLIP = (2.0, 12.0)


def quantiles(values):
    values = np.asarray([v for v in np.asarray(values, float).ravel() if np.isfinite(v)], float)
    if not len(values):
        return {"count": 0}
    q = np.quantile(values, [0, .05, .25, .5, .75, .95, 1])
    return {"count": int(len(values)), "min": float(q[0]), "p05": float(q[1]), "p25": float(q[2]), "median": float(q[3]),
            "p75": float(q[4]), "p95": float(q[5]), "max": float(q[6]), "mean": float(values.mean())}


def contour_geometry(C):
    """[a, R0, kappa, tri_top, tri_bot] of an (N,2) (R,Z) polygon, same definitions as derive._shape_from_contour."""
    R, Z = C[:, 0], C[:, 1]
    Rmax, Rmin = float(R.max()), float(R.min())
    Zmax, Zmin = float(Z.max()), float(Z.min())
    width = Rmax - Rmin
    if width <= 0:
        return np.full(len(GEOM_NAMES), np.nan)
    a = 0.5 * width
    R0 = 0.5 * (Rmax + Rmin)
    kappa = (Zmax - Zmin) / width
    tri_top = (R0 - float(R[int(np.argmax(Z))])) / a
    tri_bot = (R0 - float(R[int(np.argmin(Z))])) / a
    return np.array([a, R0, kappa, tri_top, tri_bot], float)


def extract_geometry_frames(psi, grid_R, grid_Z, machine, envelope, axis_sign):
    """(T, 5) geometry from the scorer's LCFS extraction; nan rows where extraction failed."""
    out = np.full((psi.shape[0], len(GEOM_NAMES)), np.nan)
    mask_f = envelope.astype(float)
    for t in range(psi.shape[0]):
        C = extract_lcfs(np.asarray(psi[t], np.float64), grid_R, grid_Z, machine, envelope, mask_f, axis_sign=axis_sign)
        if C is not None and len(C) >= 8:
            g = contour_geometry(C)
            if 0.8 <= g[2] <= 4.0 and np.isfinite(g).all():   # derive.PLAUSIBLE kappa range
                out[t] = g
    return out


def fill_failed_geometry(geom, fallback=None):
    """Replace nan rows by the shot median (or `fallback`); returns (geom, filled_mask, n_failed)."""
    geom = np.array(geom, float, copy=True)
    bad = ~np.isfinite(geom).all(axis=1)
    if bad.any():
        if (~bad).any():
            med = np.median(geom[~bad], axis=0)
        elif fallback is not None:
            med = np.asarray(fallback, float)
        else:
            med = np.array([0.55, 1.7, 1.75, 0.4, 0.5])
        geom[bad] = med
    return geom, bad, int(bad.sum())


def q95_inputs(geom, Ip_A, Itf_A, n_turns):
    """Physics features from geometry (T,5), |Ip| in A and |I_TF| in A."""
    a, R0, kappa, tri_top, tri_bot = (geom[:, j] for j in range(5))
    Ip = np.maximum(np.abs(np.asarray(Ip_A, float)), 1.0)
    B_T = MU0 * n_turns * np.abs(np.asarray(Itf_A, float)) / (2 * np.pi * R0)
    eps = a / R0
    delta = 0.5 * (tri_top + tri_bot)
    q_cyl = 2 * np.pi * a ** 2 * B_T * (1 + kappa ** 2) / 2 / (MU0 * R0 * Ip)
    shape89 = (1 + kappa ** 2 * (1 + 2 * delta ** 2 - 1.2 * delta ** 3)) / 2 * (1.17 - 0.65 * eps) / (1 - eps ** 2) ** 2
    q_iter89 = 5 * a ** 2 * B_T / (R0 * Ip * 1e-6) * shape89
    return {"a": a, "R0": R0, "kappa": kappa, "eps": eps, "delta": delta, "B_T": B_T, "Ip_MA": Ip * 1e-6,
            "q_cyl": q_cyl, "q_iter89": q_iter89}


def design_matrix(name, f):
    """(A, offset, coefficient names) so that log q95 = A @ c + offset."""
    n = len(f["q_cyl"])
    one = np.ones(n)
    lq, lk = np.log(f["q_cyl"]), np.log(f["kappa"])
    if name == "loglin6":
        return np.column_stack([one, lq, lk, f["eps"], f["eps"] ** 2, f["delta"]]), np.zeros(n), \
            ["c0", "c1_log_qcyl", "c2_log_kappa", "c3_eps", "c4_eps2", "c5_delta"]
    if name == "loglin_fixed":
        return np.column_stack([one, lk, f["eps"], f["delta"]]), lq, ["c0", "c2_log_kappa", "c3_eps", "c5_delta"]
    if name == "iter89":
        return np.column_stack([one, np.log(f["q_iter89"])]), np.zeros(n), ["c0", "c1_log_qiter89"]
    if name == "iter89_fixed":
        return one[:, None], np.log(f["q_iter89"]), ["c0"]
    eps = f["eps"]
    A3, names3 = np.column_stack([one, lk, f["delta"]]), ["c0", "c2_log_kappa", "c5_delta"]
    if name in PINNED_EXPONENT:
        return A3, lq - PINNED_EXPONENT[name] * np.log(1 - eps ** 2), names3
    if name == "wesson":
        return A3, lq + np.log(1 + eps ** 2), names3
    if name == "iter89_shape":
        return A3, lq + np.log((1.17 - 0.65 * eps) / (1 - eps ** 2) ** 2), names3
    raise ValueError(name)


def predict_log_q95(name, coefficients, f):
    A, offset, names = design_matrix(name, f)
    c = np.array([coefficients[k] for k in names], float)
    return A @ c + offset


def pooled_r2(y, yhat):
    """Scorer-style pooled R2: 1 - SS_res / SS_tot over every element."""
    y = np.asarray(y, float); yhat = np.asarray(yhat, float)
    tot = float(np.sum((y - y.mean()) ** 2))
    return float(1 - np.sum((y - yhat) ** 2) / tot) if tot > 0 else 0.0
