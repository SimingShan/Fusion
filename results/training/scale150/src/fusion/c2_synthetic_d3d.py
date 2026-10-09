"""DIII-D variant of the synthetic free-boundary generator (fidelity diagnostic against EFIT labels).

Runs the SAME physics as ``fusion.c2_synthetic.SyntheticMASTSolver`` (FreeGS-style profile
j = lambda[(R/R0) beta0 (1-s^ap)^gp + (R0/R)(1-beta0)(1-s^af)^gf], free-boundary Delta* solve with the
free-space Green boundary, Picard on the profile with a 2x2 Newton position controller, optional beta0 secant
for radial force balance in the REAL vacuum field) on the DIII-D geometry:

* coil tables: ``fusion.c2_fields_data.build_machine_setup("DIII-D")`` (F1A..F9B kA-turn rectangles /
  parallelograms, ECOILA with the label-calibrated effective turn count from
  data/physics_tables/diii_d_calibration.json);
* grid R 0.84-2.54, Z -1.6..1.6, scorer envelope starter/fusion_scoring/masks/d3d_envelope.npz;
* convention: work in phi = sign(Ip) * (coil flux) + plasma flux with |Ip| > 0, i.e. the axis is a MAXIMUM
  of phi; DIII-D EFIT stores psi with the axis as a MINIMUM, so phi_EFIT = -efit_psirz;
* vertical control: up/down ANTIsymmetric F7A/F7B correction dI6 (strongest radial-field lever at the
  envelope centroid: 2.08 G per kA-turn, vs 1.52 F9, 1.42 F6, 1.28 F2); radial control / force balance:
  SYMMETRIC F6A/F6B correction dI5 (strongest vertical-field lever: 5.95 G per kA-turn).

The MAST class is subclassed so that ``_controlled_solve`` (Picard + position controller) is reused verbatim;
``solve_frame`` is re-implemented with three modes (``balance``: MAST behaviour, beta0 adjusted until dI5 = 0;
``control``: fixed profile, dI5 and dI6 free; ``vertical``: fixed profile, vertical control only, radial
position free).

Also here: least-squares fit of the FreeGS profile family to an EFIT label (j = -Delta* phi / (mu0 R) inside
the official LCFS), the linear "oracle current" test (EFIT j placed in the real vacuum field), and the
exterior-residual multipole regression (gauge, uniform B_Z, quadrupole, odd terms).

Label use: DIII-D TRAIN labels only, and only for this diagnostic (profile fit, position target, scoring).
"""
from __future__ import annotations

from dataclasses import dataclass
import time as wall_time

import numpy as np
from scipy.optimize import least_squares, nnls

from fusion import c2_fields_data as cd
from fusion import c2_synthetic as cs
from fusion.mast_physics import FreeBoundaryGS, MU0

TF_TURNS_D3D = 144                     # fusion/c2_fields_data.TF_TURNS
R0_D3D = 1.59                          # m, reference radius of the profile parametrisation (envelope centroid)
Z_CTRL = ("magnetics_F7A", "magnetics_F7B")    # antisymmetric pair: vertical position
R_CTRL = ("magnetics_F6A", "magnetics_F6B")    # symmetric pair: radial position / force balance
# MAST v1 generator ranges (data/c2_synth_mast_v1/manifest.json)
MAST_V1_RANGES = {"beta0": (0.15, 0.75), "alpha_p": (1.0, 2.5), "gamma_p": (1.0, 3.0), "alpha_f": (1.2, 3.0), "gamma_f": (0.4, 1.4)}
# MAST v1 position prior: r_target ~ U(0.76, 0.90) around the envelope centroid 0.8397 m (-9.5 % .. +7.2 %),
# z_target ~ N(0, 0.02) clipped at 0.08 (2.4 % / 9.5 % of the centroid radius). Scaled to the DIII-D centroid.
MAST_V1_R_FRAC = (0.76 / 0.8397, 0.90 / 0.8397)
MAST_V1_Z_STD_FRAC = 0.02 / 0.8397
MAST_V1_Z_MAX_FRAC = 0.08 / 0.8397


def sample_centroid_prior(rng, centre):
    """(r_target, z_target) with the MAST v1 sampling rule scaled to this machine's envelope centroid."""
    Rc, Zc = centre
    r = float(rng.uniform(MAST_V1_R_FRAC[0] * Rc, MAST_V1_R_FRAC[1] * Rc))
    z = float(Zc + np.clip(rng.normal(0.0, MAST_V1_Z_STD_FRAC * Rc), -MAST_V1_Z_MAX_FRAC * Rc, MAST_V1_Z_MAX_FRAC * Rc))
    return r, z


class SyntheticD3DSolver(cs.SyntheticMASTSolver):
    """Free-boundary GS with prescribed FreeGS profiles in the real DIII-D vacuum field, F7/F6 position control."""

    def __init__(self, setup: cd.MachineSetup | None = None, R0: float = R0_D3D, lcfs_iter: int = 14,
                 z_ctrl=Z_CTRL, r_ctrl=R_CTRL, ecoil_shape_factor: float = 1.0, gs: FreeBoundaryGS | None = None):
        """``ecoil_shape_factor`` f scales the SPATIAL part of the ECOILA table (T_E -> f T_E + (1 - f) mean T_E): the
        gauge-calibrated 137-turn table reproduces the constant flux level of the E-coil but its field at the plasma
        (B_Z ~ +250 G at 1.2 MA) is not supported by the labels (see scripts/analysis/synth_fidelity_d3d.py
        calibration pass); f = 1 is the raw physics_fields model."""
        self.setup = cd.build_machine_setup("DIII-D") if setup is None else setup
        if self.setup.machine != "DIII-D":
            raise ValueError("DIII-D setup required")
        self.R, self.Z = self.setup.grid_R, self.setup.grid_Z
        self.envelope = self.setup.envelope
        self.solver = FreeBoundaryGS(self.R, self.Z) if gs is None else gs
        self.RR, self.ZZ = self.solver.RR, self.solver.ZZ
        self.dA = self.solver.dR * self.solver.dZ
        self.names = list(self.setup.table_names)
        self.matrix = np.array(self.setup.table_matrix, np.float64, copy=True)
        self.ecoil_shape_factor = float(ecoil_shape_factor)
        ie = self.names.index("magnetics_ECOILA")
        te = self.matrix[ie]
        self.ecoil_table_mean = float(te.mean())
        self.matrix[ie] = self.ecoil_shape_factor * te + (1.0 - self.ecoil_shape_factor) * self.ecoil_table_mean
        self.tables = {k: self.matrix[i].reshape(65, 65) for i, k in enumerate(self.names)}
        self.z_ctrl, self.r_ctrl = tuple(z_ctrl), tuple(r_ctrl)
        self.T6 = self.tables[z_ctrl[0]] - self.tables[z_ctrl[1]]     # Wb/rad per kA-turn antisymmetric (Z control)
        self.T5 = self.tables[r_ctrl[0]] + self.tables[r_ctrl[1]]     # Wb/rad per kA-turn symmetric (R control)
        self.R0 = float(R0)
        self.lcfs_iter = int(lcfs_iter)
        self.i6u, self.i6l = self.names.index(z_ctrl[0]), self.names.index(z_ctrl[1])
        self.i5u, self.i5l = self.names.index(r_ctrl[0]), self.names.index(r_ctrl[1])
        self.i_ecoil = self.names.index("magnetics_ECOILA")

    R_BALANCE_BOUNDS = (1.35, 2.0)      # m, admissible balanced centroid radius on DIII-D

    def vacuum(self, currents_kA):
        return (np.asarray(currents_kA, np.float64) @ self.matrix).reshape(65, 65)

    def initial_current(self, ip_A, z_target, Rc=1.7, a=0.55, b=0.9):
        j = np.maximum(0.0, 1.0 - ((self.RR - Rc) / a) ** 2 - ((self.ZZ - z_target) / b) ** 2) * self.envelope
        j[[0, -1], :] = 0.0
        j[:, [0, -1]] = 0.0
        if not j.sum() > 0:
            raise ValueError("empty_initial_current")
        return j * (ip_A / (j.sum() * self.dA))

    # ------------------------------------------------------------------ #
    def solve_frame(self, currents_kA, ip_kA, tf_kA, params: cs.ProfileParams, z_target=0.0, r_target=1.7, *,
                    mode="balance", ip_sign=1.0, initial_current=None, relaxation=0.5, max_iterations=80, tolerance=1e-3,
                    r_tolerance=3e-3, z_tolerance=3e-3, max_dI=60.0, max_seconds=120.0, balance_tol=0.5, balance_steps=6,
                    beta0_bounds=(0.02, 0.95), search_tolerance=2e-3, extra_vacuum=None, beta_p_target=None, beta_p_tol=0.03,
                    beta_p_steps=6, r_init=None):
        """Free-boundary equilibrium of the frame's inputs with a prescribed profile.

        ``currents_kA`` are the SHIPPED signed signals (table order), ``ip_kA`` the SIGNED plasma current; the solve
        runs in the axis-maximum convention phi = sign(Ip) * coil flux + plasma flux. Modes:
          balance  - MAST generator behaviour: current centroid pinned at (r_target, z_target) by dI5 (sym F6) and
                     dI6 (antisym F7); outer secant on beta0 until dI5 = 0 (radial force balance in the real field).
          control  - profile fixed (all five parameters); dI5, dI6 free and reported.
          vertical - profile fixed; only dI6 (vertical) active, the radial position is whatever the real field gives.
        v9: ``beta_p_target`` (any mode) = position-from-balance (fusion.c2_synthetic._beta_p_loop: pinned solves, beta0 secant
        for the beta_p target alternated with a secant on r_target until dI5 = 0); ``r_init`` seeds the search.
        Returns the MAST result dict (psi is phi, axis maximum; currents_kA are sign(Ip)*inputs + corrections).
        """
        started = wall_time.monotonic()
        sign = float(np.sign(ip_kA)) if ip_sign is None else float(ip_sign)
        ip_A = abs(float(ip_kA)) * 1000.0
        if not ip_A > 0:
            raise ValueError("nonzero Ip required")
        currents = sign * np.asarray(currents_kA, np.float64).copy()
        vac = self.vacuum(currents)
        if extra_vacuum is not None:
            vac = vac + np.asarray(extra_vacuum, np.float64)
        position_balance = beta_p_target is not None
        if initial_current is None:
            j = self.initial_current(ip_A, z_target, Rc=r_target if r_init is None else float(r_init))
        else:
            j = np.asarray(initial_current, np.float64).copy()
        j *= ip_A / (j.sum() * self.dA)
        radial = mode != "vertical" or position_balance
        common = dict(relaxation=relaxation, max_iterations=max_iterations, r_tolerance=r_tolerance, z_tolerance=z_tolerance,
                      max_dI=max_dI, started=started, max_seconds=max_seconds, radial=radial)
        outer, total_iterations = [], 0
        beta0 = float(params.beta0)
        prm = params.with_beta0(beta0)
        balance = mode == "balance" and not position_balance
        at_bound, beta_p_achieved = False, np.nan
        j, d5, d6, history, status, converged = self._controlled_solve(vac, ip_A, prm, j, 0.0, 0.0, r_target, z_target,
                                                                       tolerance=search_tolerance if (balance or position_balance) else tolerance, **common)
        total_iterations += len(history)
        outer.append((beta0, d5, converged))
        balanced = False
        if position_balance and converged:
            try:
                (j, d5, d6, history, status, converged, beta0, beta_p_achieved, r_target, outer_bp, at_bound, n_it,
                 balanced) = self._beta_p_loop(vac, ip_A, prm, j, d5, d6, r_target, z_target, float(beta_p_target), common,
                                               search_tolerance=search_tolerance, tolerance=tolerance, beta0_bounds=beta0_bounds,
                                               beta_p_tol=beta_p_tol, beta_p_steps=beta_p_steps, history=history, status=status,
                                               balance_tol=balance_tol)
            except ValueError as exc:
                converged, status = False, f"beta_p_loop:{exc}"
                outer_bp, n_it = [], 0
            total_iterations += n_it
            outer.extend(outer_bp)
            prm = params.with_beta0(beta0)
        elif balance and converged:
            best = (abs(d5), j.copy(), d5, d6, beta0, history, status)
            b_prev, d_prev = beta0, d5
            # dI5 (kA-turn of symmetric F6) decreases with beta0; F6 gives ~6 G/kA-turn, MAST P5 ~1 G/kA-turn,
            # so the rough first move is ~6x smaller per kA-turn than on MAST
            b_next = float(np.clip(beta0 + 0.06 * d5, *beta0_bounds))
            for _ in range(balance_steps):
                if best[0] < balance_tol or abs(b_next - b_prev) < 1e-4:
                    break
                prm = params.with_beta0(b_next)
                jt, d5t, d6t, ht, st, ct = self._controlled_solve(vac, ip_A, prm, best[1].copy(), best[2], best[3], r_target, z_target,
                                                                  tolerance=search_tolerance, **common)
                total_iterations += len(ht)
                outer.append((b_next, d5t, ct))
                if not ct:
                    break
                if abs(d5t) < best[0]:
                    best = (abs(d5t), jt.copy(), d5t, d6t, b_next, ht, st)
                if abs(d5t - d_prev) < 1e-6:
                    break
                b_new = b_next - d5t * (b_next - b_prev) / (d5t - d_prev)
                b_prev, d_prev = b_next, d5t
                b_next = float(np.clip(b_new, *beta0_bounds))
            _, j, d5, d6, beta0, history, status = best
            prm = params.with_beta0(beta0)
            at_bound = bool(beta0 <= beta0_bounds[0] + 1e-9 or beta0 >= beta0_bounds[1] - 1e-9)
            j, d5, d6, ht, status, converged = self._controlled_solve(vac, ip_A, prm, j, d5, d6, r_target, z_target,
                                                                      tolerance=tolerance, **common)
            total_iterations += len(ht)
            history = history + ht
            balanced = abs(d5) < balance_tol
        plasma = self.solver.solve(j)
        psi = vac + d5 * self.T5 + d6 * self.T6 + plasma
        try:
            j_final, lam, surface = self.profile_current(psi, prm, ip_A, n_iter=22)
            final_residual = float(np.linalg.norm(j_final - j) / max(np.linalg.norm(j), 1e-30))
            if position_balance:
                beta_p_achieved = self.beta_p_of(psi, prm, lam, surface, ip_A)
                balanced = bool(converged and abs(d5) < balance_tol and abs(beta_p_achieved - beta_p_target) < beta_p_tol * beta_p_target)
        except ValueError:
            lam, surface, final_residual, converged = np.nan, None, np.inf, False
        currents[self.i5u] += d5
        currents[self.i5l] += d5
        currents[self.i6u] += d6
        currents[self.i6l] -= d6
        return {"psi": psi, "j": j, "plasma_psi": plasma, "currents_kA": currents, "dI5_kAt": d5, "dI6_kAt": d6, "lam": lam,
                "surface": surface, "iterations": total_iterations, "residual": final_residual, "history": history, "status": status,
                "converged": bool(converged and final_residual < 3 * tolerance), "z_target": float(z_target),
                "r_target": float(r_target), "radial_free": bool(balanced), "outer": outer, "mode": mode,
                "seconds": wall_time.monotonic() - started, "vac": vac, "ip_A": ip_A, "tf_kA": float(tf_kA), "params": prm,
                "params_sampled": params, "ip_sign": sign, "position_mode": "balance" if position_balance else "pinned",
                "beta0_at_bound": bool(at_bound), "beta_p_target": float(beta_p_target) if position_balance else np.nan,
                "beta_p_achieved": float(beta_p_achieved)}

    # ------------------------------------------------------------------ #
    def scalars(self, result):
        """q95, betaN, beta_p and shape diagnostics of the final phi (DIII-D TF turns; official derived scalars)."""
        import sys
        from pathlib import Path
        scorer = Path(cs.SCORER)
        if str(scorer) not in sys.path:
            sys.path.insert(0, str(scorer))
        from contour import encloses_point, find_contours, grid_to_physical, is_closed
        from derive import _bilinear, derive_frame
        from lcfs import extract_lcfs
        psi, params, lam, ip_A = result["psi"], result["params"], result["lam"], result["ip_A"]
        if result["surface"] is None:
            raise ValueError("no_surface")
        C, (iz, ir), axis_val, edge, inside = result["surface"]
        delta = axis_val - edge
        s = np.clip((axis_val - psi) / delta, 0.0, 1.0)
        Ip_int = cs.ShapeIntegral(params.alpha_p, params.gamma_p)
        If_int = cs.ShapeIntegral(params.alpha_f, params.gamma_f)
        pressure = lam * params.beta0 * delta / self.R0 * Ip_int(s) * inside
        if params.edge_amp > 0:      # v9 pedestal pressure (default off)
            pressure = pressure + lam * params.edge_amp * delta / self.R0 * params.edge_pressure_integral(s) * inside
        Fvac = abs(MU0 / (2.0 * np.pi) * TF_TURNS_D3D * result["tf_kA"] * 1000.0)

        def F_squared(level):
            return Fvac ** 2 + 2.0 * MU0 * lam * self.R0 * (1.0 - params.beta0) * delta * If_int(level)
        Fmin = float(np.min(F_squared(np.linspace(0, 1, 1001)))) if Fvac > 0 else np.nan
        q95 = np.nan
        if Fvac > 0 and Fmin > 0:
            level = edge + 0.05 * delta
            dz, dr = np.gradient(psi, self.Z, self.R)
            bp = np.hypot(dz, dr) / self.RR
            for curve in find_contours(psi, level):
                if is_closed(curve) and encloses_point(curve, iz, ir):
                    physical = grid_to_physical(curve, self.R, self.Z)
                    local_bp = _bilinear(bp, self.R, self.Z, physical[:, 0], physical[:, 1])
                    if np.any(local_bp <= 0):
                        break
                    integrand = 1.0 / (physical[:, 0] ** 2 * local_bp)
                    dl = np.linalg.norm(np.roll(physical, -1, axis=0) - physical, axis=1)
                    q95 = float(np.sqrt(F_squared(0.95)) / (2.0 * np.pi) * np.sum(0.5 * (integrand + np.roll(integrand, -1)) * dl))
                    break
        mean_p = float(np.sum(pressure[inside] * self.RR[inside]) / np.sum(self.RR[inside]))
        Rgeo = 0.5 * (C[:, 0].max() + C[:, 0].min())
        minor = 0.5 * np.ptp(C[:, 0])
        Bt = Fvac / Rgeo if Fvac > 0 else np.nan
        betaN = float(100.0 * 2.0 * MU0 * mean_p / Bt ** 2 * minor * Bt / (ip_A / 1e6)) if Fvac > 0 else np.nan
        lp = float(np.sum(np.linalg.norm(np.roll(C, -1, axis=0) - C, axis=1)))
        beta_p = float(2.0 * MU0 * mean_p / (MU0 * ip_A / lp) ** 2)
        C_off = extract_lcfs(psi, self.R, self.Z, "DIII-D", self.envelope, self.envelope.astype(float), axis_sign=1.0)
        derived = derive_frame(psi, self.R, self.Z, "DIII-D", self.envelope, self.envelope.astype(float), contour=C_off, axis_sign=1.0)
        envelope_on_C = _bilinear(self.envelope.astype(float), self.R, self.Z, C[:, 0], C[:, 1])
        contact = float(np.mean(envelope_on_C < 0.75))
        grid_touch = bool(((C[:, 0] < self.R[1]) | (C[:, 0] > self.R[-2]) | (C[:, 1] < self.Z[1]) | (C[:, 1] > self.Z[-2])).any())
        return {"q95": q95, "betaN": betaN, "beta_p": beta_p, "pressure_axis_Pa": float(pressure.max()), "pressure_mean_Pa": mean_p,
                "Fvac_Tm": Fvac, "F_squared_min": Fmin, "Bt_geo_T": float(Bt), "derived": derived, "lcfs_official": C_off,
                "lcfs_Rmin": float(C[:, 0].min()), "lcfs_Rmax": float(C[:, 0].max()), "lcfs_Zmin": float(C[:, 1].min()),
                "lcfs_Zmax": float(C[:, 1].max()), "envelope_contact_fraction": contact, "grid_touch": grid_touch,
                "psi_axis": axis_val, "psi_edge": edge, "psi_ptp": float(psi.max() - psi.min()), "delta": delta}

    # ------------------------------------------------------------------ #
    def label_current(self, phi, surface=None):
        """EFIT toroidal current density j = -Delta* phi / (mu0 R) inside the official LCFS of the label phi (axis max)."""
        surface = cs.plasma_surface(phi, self.R, self.Z, self.envelope, n_iter=22) if surface is None else surface
        if surface is None:
            raise ValueError("label_has_no_closed_surface")
        j = -(self.solver.matrix @ phi.ravel()).reshape(self.solver.shape) / (MU0 * self.RR)
        j[~surface[4]] = 0.0
        j.ravel()[self.solver.boundary] = 0.0
        return j, surface

    def oracle_solve(self, currents_kA, ip_kA, phi_label, ip_sign=1.0, renormalise=True):
        """Linear test of the vacuum/coil model: the EFIT current (optionally renormalised to the measured Ip) is
        placed in the REAL coil field: phi = sign(Ip) * coil + G j_EFIT. No profile assumption, no control."""
        sign = float(ip_sign)
        ip_A = abs(float(ip_kA)) * 1000.0
        j, surface = self.label_current(phi_label)
        ip_label = float(j.sum() * self.dA)
        if renormalise and ip_label > 0:
            j = j * (ip_A / ip_label)
        vac = self.vacuum(sign * np.asarray(currents_kA, np.float64))
        plasma = self.solver.solve(j)
        return {"psi": vac + plasma, "vac": vac, "plasma_psi": plasma, "j": j, "ip_label_A": ip_label, "ip_A": ip_A,
                "label_surface": surface}


# ---------------------------------------------------------------------- #
# Profile fit of the FreeGS family to an EFIT label
# ---------------------------------------------------------------------- #
@dataclass
class ProfileFit:
    params: cs.ProfileParams
    lam: float
    rel_rmse: float             # weighted relative RMS of the j fit inside the LCFS
    rel_rmse_unweighted: float
    ip_label_A: float
    ip_ratio: float             # label-implied Ip / measured |Ip|
    n_points: int
    starts: int
    best_cost: float
    centroid_R: float           # current centroid of the LABEL j (control target)
    centroid_Z: float
    axis_R: float
    axis_Z: float
    beta_p_label: float         # FreeGS closure beta_p of the fitted profile at the label flux
    j_axis_ratio: float         # fitted / label j at the axis cell (peakedness check)


def fit_freegs_profile(phi, solver: SyntheticD3DSolver, ip_A, R0=None, s_range=(0.02, 0.98),
                       bounds=((0.4, 0.3, 0.4, 0.3), (6.0, 6.0, 6.0, 6.0)),
                       starts=((1.5, 2.0, 1.5, 1.0), (1.0, 1.0, 2.0, 0.6), (2.5, 3.0, 1.0, 2.0), (1.2, 1.5, 3.0, 0.4))) -> ProfileFit:
    """Variable-projection least squares of j_EFIT on the FreeGS family (NNLS amplitudes, bounded shape params).

    Shell weights 1/sqrt(count) per s-shell (12 shells) equalise the radial coverage as in
    fusion.mast_profiles.fit_profile_target. lambda is then re-derived from the MEASURED Ip (as the generator does),
    so only beta0 and the four shape exponents are carried into the solve.
    """
    R0 = solver.R0 if R0 is None else float(R0)
    j, surface = solver.label_current(phi)
    C, (iz, ir), axis_val, edge, inside = surface
    s = np.clip((axis_val - phi) / (axis_val - edge), 0.0, 1.0)
    sel = inside & (s > s_range[0]) & (s < s_range[1])
    sel.ravel()[solver.solver.boundary] = False
    if sel.sum() < 40:
        raise ValueError("too_few_label_points")
    shell = np.minimum((s[sel] * 12).astype(int), 11)
    w = 1.0 / np.sqrt(np.bincount(shell, minlength=12)[shell])
    y = j[sel]
    rr, ss = solver.RR[sel], s[sel]

    def basis(theta):
        ap, gp, af, gf = theta
        return np.stack([(rr / R0) * cs.shape_function(ss, ap, gp), (R0 / rr) * cs.shape_function(ss, af, gf)], axis=1)

    def amplitudes(theta):
        B = basis(theta)
        a, _ = nnls(B * w[:, None], y * w)
        return a, B

    def resid(theta):
        a, B = amplitudes(theta)
        return (B @ a - y) * w

    best = None
    for x0 in starts:
        try:
            sol = least_squares(resid, np.asarray(x0, float), bounds=bounds, max_nfev=400, xtol=1e-6, ftol=1e-8)
        except Exception:  # noqa: BLE001
            continue
        if best is None or sol.cost < best.cost:
            best = sol
    if best is None:
        raise ValueError("profile_fit_failed")
    a, B = amplitudes(best.x)
    total = float(a.sum())
    if not total > 0:
        raise ValueError("degenerate_profile_fit")
    beta0 = float(np.clip(a[0] / total, 0.0, 1.0))
    fitted = B @ a
    rel = float(np.linalg.norm((fitted - y) * w) / max(np.linalg.norm(y * w), 1e-30))
    rel_u = float(np.linalg.norm(fitted - y) / max(np.linalg.norm(y), 1e-30))
    ip_label = float(j.sum() * solver.dA)
    params = cs.ProfileParams(beta0, float(best.x[0]), float(best.x[1]), float(best.x[2]), float(best.x[3]))
    # closure beta_p of the fitted profile evaluated on the label flux with lambda from the measured Ip
    shape = inside * ((solver.RR / R0) * beta0 * cs.shape_function(s, params.alpha_p, params.gamma_p)
                      + (R0 / solver.RR) * (1 - beta0) * cs.shape_function(s, params.alpha_f, params.gamma_f))
    lam = ip_A / max(float(shape.sum() * solver.dA), 1e-30)
    delta = axis_val - edge
    pressure = lam * beta0 * delta / R0 * cs.ShapeIntegral(params.alpha_p, params.gamma_p)(s) * inside
    mean_p = float(np.sum(pressure[inside] * solver.RR[inside]) / np.sum(solver.RR[inside]))
    lp = float(np.sum(np.linalg.norm(np.roll(C, -1, axis=0) - C, axis=1)))
    beta_p = float(2.0 * MU0 * mean_p / (MU0 * ip_A / lp) ** 2)
    jl = j.sum()
    cR = float(np.sum(j * solver.RR) / jl) if jl else np.nan
    cZ = float(np.sum(j * solver.ZZ) / jl) if jl else np.nan
    j_axis_fit = float(lam * shape[iz, ir])
    return ProfileFit(params, float(a.sum()), rel, rel_u, ip_label, ip_label / ip_A, int(sel.sum()), len(starts), float(best.cost),
                      cR, cZ, float(solver.R[ir]), float(solver.Z[iz]), beta_p, j_axis_fit / max(float(j[iz, ir]), 1e-30))


# ---------------------------------------------------------------------- #
# Exterior residual multipole regression (what external field is "missing")
# ---------------------------------------------------------------------- #
MULTIPOLE_NAMES = ["gauge", "uniform_Bz_T", "quadrupole", "odd_Z", "odd_R2Z"]


def multipole_basis(RR, ZZ):
    """Homogeneous Delta* solutions: 1 (gauge), R^2/2 (uniform B_Z, coefficient = B_Z in T), (R^4 - 4 R^2 Z^2)/8
    (quadrupole; B_Z = (R^2 - 2 Z^2)/2 per unit coefficient), Z and R^2 Z (odd, radial-field terms)."""
    return np.stack([np.ones_like(RR), RR ** 2 / 2, (RR ** 4 - 4 * RR ** 2 * ZZ ** 2) / 8, ZZ, RR ** 2 * ZZ], axis=-1)


def exterior_multipole_fit(phi_label, phi_pred, solver, label_surface, pred_surface, margin_cells=2):
    """Regress phi_label - phi_pred outside BOTH plasmas (dilated by margin_cells) on the multipole basis, stage-wise.

    Returns coefficients (full basis), residual RMS after each cumulative stage (raw, +gauge, +uniform B_Z,
    +quadrupole, +odd), and the correction field of the full fit and of the uniform-B_Z-only fit (gauge excluded).
    """
    from scipy.ndimage import binary_dilation
    inside = label_surface[4] | (pred_surface[4] if pred_surface is not None else False)
    region = ~binary_dilation(inside, iterations=margin_cells)
    region[[0, -1], :] = False
    region[:, [0, -1]] = False
    r = (phi_label - phi_pred)[region]
    B = multipole_basis(solver.RR, solver.ZZ)[region]
    out = {"rms_raw": float(np.sqrt(np.mean(r ** 2))), "n_points": int(region.sum())}
    stages = [1, 2, 3, 5]
    coef_full = None
    for k in stages:
        c, *_ = np.linalg.lstsq(B[:, :k], r, rcond=None)
        out[f"rms_after_{k}"] = float(np.sqrt(np.mean((r - B[:, :k] @ c) ** 2)))
        if k == 2:
            coef_bz = c
        coef_full = c
    out["coef"] = {n: float(coef_full[i]) for i, n in enumerate(MULTIPOLE_NAMES)}
    out["uniform_Bz_T_bzonly"] = float(coef_bz[1])
    full_basis = multipole_basis(solver.RR, solver.ZZ)
    out["correction_full"] = full_basis[..., 1:] @ coef_full[1:]          # gauge excluded (does not move the plasma)
    out["correction_bz"] = full_basis[..., 1] * coef_bz[1]
    return out


def decay_index(phi_vac, R, Z, r, z):
    """n = -(R/B_Z) dB_Z/dR of an external field at (r, z), B_Z = (1/R) dphi/dR (central differences on the grid)."""
    RR = np.asarray(R, float)
    iz = int(np.argmin(np.abs(np.asarray(Z, float) - z)))
    ir = int(np.clip(np.argmin(np.abs(RR - r)), 1, len(RR) - 2))
    row = np.asarray(phi_vac, float)[iz]
    bz = np.gradient(row, RR) / RR
    dbz = np.gradient(bz, RR)
    with np.errstate(divide="ignore", invalid="ignore"):
        n = -RR[ir] * dbz[ir] / bz[ir]
    return float(n), float(bz[ir])


def coil_model_calibration(solver: "SyntheticD3DSolver", currents_kA, ip_kA, phi_label, ip_sign=1.0, margin_cells=2):
    """Per-frame linear check of the coil model against the label (label j placed in the grid, exterior regression).

    Regresses the label-implied external flux phi_label - G j_EFIT on [F-coil field, E-coil SPATIAL field, 1] outside
    the (dilated) LCFS -> factors f_F, f_E (1 = raw model); and the vertical-field budget at the EFIT axis:
    B_Z(F) + f_E_axis * B_Z(E, spatial) + B_Z(plasma) = 0 -> f_E_axis. Uses the RAW tables regardless of the solver's
    ecoil_shape_factor.
    """
    from scipy.ndimage import binary_dilation
    sign = float(ip_sign)
    cur = sign * np.asarray(currents_kA, np.float64)
    j, surface = solver.label_current(phi_label)
    plasma = solver.solver.solve(j)
    ie = solver.names.index("magnetics_ECOILA")
    raw = solver.setup.table_matrix
    F = sum(cur[i] * raw[i].reshape(65, 65) for i in range(len(solver.names)) if i != ie)
    E_sp = cur[ie] * (raw[ie].reshape(65, 65) - raw[ie].mean())
    inside = surface[4]
    ext = ~binary_dilation(inside, iterations=margin_cells)
    ext[[0, -1], :] = False
    ext[:, [0, -1]] = False
    y = (phi_label - plasma)[ext]
    X = np.stack([F[ext], E_sp[ext], np.ones(ext.sum())], axis=1)
    c, *_ = np.linalg.lstsq(X, y, rcond=None)
    rms_before = float(np.sqrt(np.mean((y - np.stack([F[ext], E_sp[ext], np.ones(ext.sum())], 1) @ np.array([1.0, 1.0, c[2]])) ** 2)))
    rms_after = float(np.sqrt(np.mean((y - X @ c) ** 2)))
    # axis budget
    _, _, iz, ir = None, None, surface[1][0], surface[1][1]
    R = solver.R

    def bz(f):
        return float((np.gradient(f, R, axis=1) / solver.RR)[iz, ir])
    bF, bE, bP, bL = bz(F), bz(E_sp), bz(plasma), bz(phi_label)
    f_axis = -(bF + bP - bL) / bE if abs(bE) > 1e-9 else np.nan
    return {"cal_f_F": float(c[0]), "cal_f_E": float(c[1]), "cal_gauge": float(c[2]), "cal_rms_raw": rms_before, "cal_rms_fit": rms_after,
            "cal_Bz_F_G": bF * 1e4, "cal_Bz_E_G": bE * 1e4, "cal_Bz_plasma_G": bP * 1e4, "cal_Bz_label_G": bL * 1e4,
            "cal_Bz_residual_raw_G": (bF + bE + bP - bL) * 1e4, "cal_f_E_axis": float(f_axis), "cal_n_ext": int(ext.sum())}
