"""Synthetic free-boundary MAST equilibria on real public inputs (Challenge 2 training augmentation).

For a REAL MAST public-test frame (its coil currents, solenoid current, Ip and TF current) the
free-boundary Grad-Shafranov equation

    Delta* psi = -mu0 R j_phi,        psi = psi_coils + psi_plasma  (free-space Green boundary)

is solved with the validated ``fusion.mast_physics.FreeBoundaryGS`` operator and a PRESCRIBED
FreeGS-style current profile inside the last closed flux surface

    j_phi = lambda [ (R/R0) beta0 (1 - s^alpha_p)^gamma_p + (R0/R) (1 - beta0) (1 - s^alpha_f)^gamma_f ]
    s = (psi_axis - psi) / (psi_axis - psi_edge)  in [0, 1]  (0 on axis, 1 at the LCFS),

lambda fixed so that the integral equals the frame's Ip. This is exactly j = R p' + F F'/(mu0 R) with

    p'(psi)   = lambda beta0 / R0 (1 - s^alpha_p)^gamma_p            -> p(s)   = lambda beta0 Delta / R0 * I_p(s)
    F F'(psi) = mu0 lambda R0 (1 - beta0) (1 - s^alpha_f)^gamma_f    -> F^2(s) = F_vac^2 + 2 mu0 lambda R0 (1-beta0) Delta I_f(s)

with I(s) = int_s^1 (1 - t^alpha)^gamma dt and Delta = psi_axis - psi_edge, so q95 and betaN follow
from the same closure with ``fusion.profile_scalars`` conventions (TF: 24 turns, F_vac = mu0/(2 pi) N |I_TF|,
volume-averaged pressure, betaN = 100 * 2 mu0 <p> a / (Bt Ip[MA])).

The plasma is free-boundary: its position and shape follow from the REAL vacuum field. Elongated MAST
plasmas are vertically unstable with fixed coil currents, so the fixed-point iteration would diverge in
the vertical mode. MAST's vertical position was controlled with the P6 (radial-field) coils, and we do
the same: an antisymmetric P6 correction dI6 (p6u += dI6, p6l -= dI6, kA-turn) is solved each Picard
step (Newton on the new current centroid) so that the converged equilibrium has its current centroid at a
sampled Z target. The stored record carries the ADJUSTED P6 currents, so the inputs seen by the network
are exactly the ones the equilibrium was solved in (a genuine GS solution of the stored inputs).

Sign conventions: every MAST public frame has Ip > 0; positive currents give positive psi (psi = R A_phi,
Wb/rad) so the axis is a MAXIMUM, the stored MAST target convention. The coil response tables are the
ones the CNN featurizer uses (``fusion.c2_fields_data.MachineSetup.table_matrix``: P-coils are total
kA-turn over N rectangles, solenoid is kA feed x 656 turns), so channel 1 of the network is the exact
vacuum field of the synthetic equilibrium.

Label use: none. No MAST labels exist locally; only public inputs are read.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import sys
import time as wall_time

import numpy as np

from fusion import c2_fields_data as cd
from fusion import physics_fields as pf
from fusion.mast_physics import FreeBoundaryGS, MU0

ROOT = Path(__file__).resolve().parents[1]
SCORER = ROOT / "starter" / "fusion_scoring"
if str(SCORER) not in sys.path:
    sys.path.insert(0, str(SCORER))
from contour import encloses_point, find_contours, grid_to_physical, is_closed, resample_closed  # noqa: E402
from derive import _bilinear, _points_in_polygon, derive_frame, internal_inductance  # noqa: E402
from lcfs import _best_contour_at, extract_lcfs  # noqa: E402
from o_point import find_o_point  # noqa: E402

TF_TURNS_MAST = 24
R0_DEFAULT = 0.85           # m, fixed reference radius of the FreeGS-style parametrisation
P6U, P6L = "magnetics_p6u_current", "magnetics_p6l_current"
P5U, P5L = "magnetics_p5u_current", "magnetics_p5l_current"
PARAM_NAMES = ["beta0", "alpha_p", "gamma_p", "alpha_f", "gamma_f"]
# Profile-parameter sampling ranges (uniform). Tuned on the pilot (see generate_synthetic_mast.py --report).
DEFAULT_RANGES = {"beta0": (0.15, 0.75), "alpha_p": (1.0, 2.5), "gamma_p": (1.0, 3.0),
                  "alpha_f": (1.0, 2.5), "gamma_f": (0.8, 2.5)}
Z_TARGET_STD = 0.03         # m, sampled current-centroid target (MAST plasmas are controlled near Z = 0)
Z_TARGET_MAX = 0.08


# --------------------------------------------------------------------------- #
# Profile functions
# --------------------------------------------------------------------------- #
def shape_function(s, alpha, gamma):
    s = np.clip(np.asarray(s, np.float64), 0.0, 1.0)
    return (1.0 - s ** alpha) ** gamma


class ShapeIntegral:
    """I(s) = int_s^1 (1 - t^alpha)^gamma dt on a fine table (Simpson-grade accuracy via dense trapezoid)."""

    def __init__(self, alpha, gamma, n=4001):
        t = np.linspace(0.0, 1.0, n)
        f = shape_function(t, alpha, gamma)
        cum = np.concatenate([[0.0], np.cumsum(0.5 * (f[1:] + f[:-1]) * np.diff(t))])
        self.t, self.tail = t, cum[-1] - cum        # int_s^1

    def __call__(self, s):
        return np.interp(np.clip(np.asarray(s, np.float64), 0.0, 1.0), self.t, self.tail)


@dataclass
class ProfileParams:
    beta0: float
    alpha_p: float
    gamma_p: float
    alpha_f: float
    gamma_f: float
    # v9 (default off): H-mode-like edge/pedestal current, j_shape += edge_amp * exp(-(1 - s) / edge_w) * (R / R0),
    # attributed to p' (an edge pressure pedestal of height lambda Delta edge_amp edge_w / R0). edge_amp = 0 -> v1 family.
    edge_amp: float = 0.0
    edge_w: float = 0.05

    def as_array(self):
        return np.asarray([self.beta0, self.alpha_p, self.gamma_p, self.alpha_f, self.gamma_f], np.float64)

    @classmethod
    def sample(cls, rng, ranges=None):
        ranges = DEFAULT_RANGES if ranges is None else ranges
        return cls(*[float(rng.uniform(*ranges[k])) for k in PARAM_NAMES])

    def with_beta0(self, beta0):
        return ProfileParams(float(beta0), self.alpha_p, self.gamma_p, self.alpha_f, self.gamma_f, self.edge_amp, self.edge_w)

    def edge_shape(self, s):
        """Edge current shape exp(-(1 - s) / w) (1 at the LCFS, decaying inward)."""
        return np.exp(-(1.0 - np.clip(np.asarray(s, np.float64), 0.0, 1.0)) / self.edge_w)

    def edge_pressure_integral(self, s):
        """int_s^1 exp(-(1 - t) / w) dt = w (1 - exp(-(1 - s) / w)) (pedestal pressure shape, 0 at the LCFS)."""
        return self.edge_w * (1.0 - self.edge_shape(s))


# --------------------------------------------------------------------------- #
# Surface extraction (scorer machinery, configurable bisection depth)
# --------------------------------------------------------------------------- #
def plasma_surface(psi, R, Z, envelope, n_iter=14, n_points=256):
    """(contour(R,Z), (iz, ir), psi_axis, psi_edge, inside-mask) of the LCFS of an axis-maximum psi, or None."""
    envelope_f = envelope.astype(np.float64)
    _, _, iz, ir, _ = find_o_point(psi, R, Z, envelope)
    if not envelope[iz, ir]:
        return None
    axis_val = float(psi[iz, ir])
    border = np.concatenate([psi[0, :], psi[-1, :], psi[:, 0], psi[:, -1]])
    lo, hi = float(border.min()), axis_val - 1e-9
    if not hi > lo:
        return None
    best = _best_contour_at(psi, hi, iz, ir, envelope_f)
    if best is None:
        return None
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        c = _best_contour_at(psi, mid, iz, ir, envelope_f)
        if c is not None:
            hi, best = mid, c
        else:
            lo = mid
    C = resample_closed(grid_to_physical(best, R, Z), n_points)
    edge = float(np.mean(_bilinear(psi, R, Z, C[:, 0], C[:, 1])))
    delta = axis_val - edge
    if delta <= 1e-8:
        return None
    RR, ZZ = np.meshgrid(R, Z)
    inside = _points_in_polygon(RR.ravel(), ZZ.ravel(), C).reshape(psi.shape)
    inside[[0, -1], :] = False
    inside[:, [0, -1]] = False
    if inside.sum() < 12:
        return None
    return C, (iz, ir), axis_val, edge, inside


# --------------------------------------------------------------------------- #
# Solver
# --------------------------------------------------------------------------- #
class SyntheticMASTSolver:
    """Free-boundary GS with prescribed profiles in the real MAST vacuum field, P6 vertical control."""

    def __init__(self, setup: cd.MachineSetup | None = None, R0: float = R0_DEFAULT, lcfs_iter: int = 14):
        self.setup = cd.build_machine_setup("MAST") if setup is None else setup
        if self.setup.machine != "MAST":
            raise ValueError("MAST setup required")
        self.R, self.Z = self.setup.grid_R, self.setup.grid_Z
        self.envelope = self.setup.envelope
        self.solver = FreeBoundaryGS(self.R, self.Z)
        self.RR, self.ZZ = self.solver.RR, self.solver.ZZ
        self.dA = self.solver.dR * self.solver.dZ
        self.names = list(self.setup.table_names)
        self.tables = {k: self.setup.table_matrix[i].reshape(65, 65) for i, k in enumerate(self.names)}
        self.T6 = self.tables[P6U] - self.tables[P6L]           # Wb/rad per kA-turn of antisymmetric P6 (Z control)
        self.T5 = self.tables[P5U] + self.tables[P5L]           # Wb/rad per kA-turn of symmetric P5 (R control)
        self.R0 = float(R0)
        self.lcfs_iter = int(lcfs_iter)
        self.i6u, self.i6l = self.names.index(P6U), self.names.index(P6L)
        self.i5u, self.i5l = self.names.index(P5U), self.names.index(P5L)

    # ---- pieces -------------------------------------------------------- #
    def vacuum(self, currents_kA):
        return (np.asarray(currents_kA, np.float64) @ self.setup.table_matrix).reshape(65, 65)

    def initial_current(self, ip_A, z_target, Rc=0.9, a=0.5, b=0.95):
        j = np.maximum(0.0, 1.0 - ((self.RR - Rc) / a) ** 2 - ((self.ZZ - z_target) / b) ** 2) * self.envelope
        j[[0, -1], :] = 0.0
        j[:, [0, -1]] = 0.0
        return j * (ip_A / (j.sum() * self.dA))

    def profile_current(self, psi, params: ProfileParams, ip_A, surface=None, n_iter=None):
        """j = J(psi): prescribed profile inside the LCFS of psi, normalised to Ip. Raises ValueError if no LCFS."""
        surface = plasma_surface(psi, self.R, self.Z, self.envelope, self.lcfs_iter if n_iter is None else n_iter) \
            if surface is None else surface
        if surface is None:
            raise ValueError("no_closed_plasma_surface")
        C, axis, axis_val, edge, inside = surface
        s = np.clip((axis_val - psi) / (axis_val - edge), 0.0, 1.0)
        shape = inside * ((self.RR / self.R0) * params.beta0 * shape_function(s, params.alpha_p, params.gamma_p)
                          + (self.R0 / self.RR) * (1.0 - params.beta0) * shape_function(s, params.alpha_f, params.gamma_f))
        if params.edge_amp > 0:      # v9 edge/pedestal current (default off)
            shape = shape + inside * ((self.RR / self.R0) * params.edge_amp * params.edge_shape(s))
        total = shape.sum() * self.dA
        if not total > 0:
            raise ValueError("empty_profile_support")
        lam = ip_A / total
        return lam * shape, lam, surface

    @staticmethod
    def centroid_z(j, ZZ):
        return float(np.sum(j * ZZ) / max(np.sum(j), 1e-30))

    @staticmethod
    def centroid_r(j, RR):
        return float(np.sum(j * RR) / max(np.sum(j), 1e-30))

    # ---- v9 helpers: pressure field with the optional pedestal, closure beta_p of a state ------------------- #
    def pressure_field(self, s, params: ProfileParams, lam, delta, inside):
        """p(s) of the profile family (Pa): lam beta0 Delta / R0 I_p(s) (+ the v9 pedestal term when edge_amp > 0)."""
        p = lam * params.beta0 * delta / self.R0 * ShapeIntegral(params.alpha_p, params.gamma_p)(s) * inside
        if params.edge_amp > 0:
            p = p + lam * params.edge_amp * delta / self.R0 * params.edge_pressure_integral(s) * inside
        return p

    def beta_p_of(self, psi, params: ProfileParams, lam, surface, ip_A):
        """beta_p = 2 mu0 <p> / (mu0 Ip / l_p)^2 (same normalisation as ``scalars``) of a state, TF-independent."""
        C, _, axis_val, edge, inside = surface
        delta = axis_val - edge
        s = np.clip((axis_val - psi) / delta, 0.0, 1.0)
        pressure = self.pressure_field(s, params, lam, delta, inside)
        mean_p = float(np.sum(pressure[inside] * self.RR[inside]) / np.sum(self.RR[inside]))
        lp = float(np.sum(np.linalg.norm(np.roll(C, -1, axis=0) - C, axis=1)))
        return float(2.0 * MU0 * mean_p / (MU0 * ip_A / lp) ** 2)

    def _beta_p_state(self, vac, ip_A, params, j, d5, d6):
        plasma = self.solver.solve(j)
        psi = vac + d5 * self.T5 + d6 * self.T6 + plasma
        _, lam, surface = self.profile_current(psi, params, ip_A)
        return self.beta_p_of(psi, params, lam, surface, ip_A)

    R_BALANCE_BOUNDS = (0.55, 1.15)     # m, admissible balanced current-centroid radius (MAST); overridden per machine
    DI5_DR_SIGN = 1.0                   # sign of d(dI5)/d(r_target) (positive on MAST P5 and DIII-D F6, both checked)

    def _beta_p_loop(self, vac, ip_A, params, j, d5, d6, r_target, z_target, beta_p_target, common, *, search_tolerance, tolerance,
                     beta0_bounds, beta_p_tol, beta_p_steps, history, status, balance_tol=0.5, r_steps=8, r_bounds=None, rounds=3):
        """v9 position-from-balance with the PINNED controller (radial + vertical control, ``common['radial']`` True):

        A. secant on beta0 until the closure beta_p of the equilibrium pinned at the current r_target equals ``beta_p_target``;
        B. secant on r_target until the symmetric radial-control current dI5 vanishes (|dI5| < ``balance_tol`` kA-turn), i.e.
           the REAL vacuum field holds the plasma at that radius (the centroid is iterated, not sampled);
        C. A and B alternated (``rounds``), then a tight warm-started solve at the balanced (beta0, r_target).
        Every solve is a warm-started pinned solve (the free-position Picard iteration drifts on ~1/3 of the frames).
        Returns (j, d5, d6, history, status, converged, beta0, beta_p, r_target, outer, at_bound, iterations, balanced)."""
        r_bounds = self.R_BALANCE_BOUNDS if r_bounds is None else r_bounds
        beta0 = float(params.beta0)
        state = dict(j=j.copy(), d5=d5, d6=d6, history=history, status=status)
        outer, iterations = [], 0

        def pinned(b, r, warm):
            nonlocal iterations
            prm = params.with_beta0(b)
            jt, d5t, d6t, ht, st, ct = self._controlled_solve(vac, ip_A, prm, warm["j"].copy(), warm["d5"], warm["d6"], r, z_target,
                                                              tolerance=search_tolerance, **common)
            iterations += len(ht)
            bp = self._beta_p_state(vac, ip_A, prm, jt, d5t, d6t) if ct else np.nan
            outer.append((b, d5t, ct, r, bp))
            return dict(j=jt, d5=d5t, d6=d6t, history=ht, status=st, converged=ct, beta_p=bp, beta0=b, r=r)

        bp = self._beta_p_state(vac, ip_A, params, j, d5, d6)
        state.update(converged=True, beta_p=bp, beta0=beta0, r=float(r_target))
        outer.append((beta0, d5, True, float(r_target), bp))
        balanced = False
        for _ in range(int(rounds)):
            # ---- A: beta0 for the beta_p target at the current radius ------------------------------------------ #
            b_prev, f_prev = state["beta0"], state["beta_p"] - beta_p_target
            b_next = float(np.clip(state["beta0"] * beta_p_target / max(state["beta_p"], 1e-6), *beta0_bounds))
            for _ in range(int(beta_p_steps)):
                if abs(f_prev) < beta_p_tol * beta_p_target or abs(b_next - b_prev) < 1e-4:
                    break
                cand = pinned(b_next, state["r"], state)
                if not cand["converged"]:
                    break
                f_t = cand["beta_p"] - beta_p_target
                if abs(f_t) < abs(f_prev) or abs(f_t) < beta_p_tol * beta_p_target:
                    state = cand
                if abs(f_t - f_prev) < 1e-9:
                    break
                b_new = b_next - f_t * (b_next - b_prev) / (f_t - f_prev)
                b_prev, f_prev = b_next, f_t
                b_next = float(np.clip(b_new, *beta0_bounds))
            # ---- B: radius for dI5 = 0 at this beta0 ---------------------------------------------------------- #
            r_prev, d_prev = state["r"], state["d5"]
            if abs(d_prev) < balance_tol:
                balanced = True
            else:
                # dI5(r_target) is monotonic (DIII-D and MAST: dI5 INCREASES with r, DI5_DR_SIGN = +1, i.e. a positive
                # dI5 means the balanced radius lies inward); first probe 3 cm towards the root, then secant on
                # dI5(r_target) with a 10 cm trust region; an unconverged probe halves the step (twice) before giving up
                step = -0.03 * np.sign(d_prev) * self.DI5_DR_SIGN
                r_next = float(np.clip(state["r"] + step, *r_bounds))
                halvings = 0
                for _ in range(int(r_steps)):
                    if abs(r_next - state["r"]) < 5e-4:
                        break
                    cand = pinned(state["beta0"], r_next, state)
                    if not cand["converged"]:
                        halvings += 1
                        if halvings > 2:
                            break
                        r_next = float(state["r"] + 0.5 * (r_next - state["r"]))
                        continue
                    if abs(cand["d5"]) < abs(state["d5"]):
                        state = cand
                    if abs(cand["d5"]) < balance_tol:
                        balanced = True
                        break
                    if abs(cand["d5"] - d_prev) < 1e-6 or abs(r_next - r_prev) < 1e-6:
                        break
                    slope = (cand["d5"] - d_prev) / (r_next - r_prev)
                    r_prev, d_prev = r_next, cand["d5"]
                    if not slope * self.DI5_DR_SIGN > 0:      # wrong-signed local slope: fall back to a 5 cm move towards the root
                        r_new = r_next - 0.05 * np.sign(cand["d5"]) * self.DI5_DR_SIGN
                    else:
                        r_new = r_next - cand["d5"] / slope
                    r_new = float(np.clip(r_new, r_next - 0.15, r_next + 0.15))
                    r_next = float(np.clip(r_new, *r_bounds))
            if not balanced:        # no balanced radius within the probe budget: give up (the record is rejected as unbalanced)
                break
            if abs(state["beta_p"] - beta_p_target) < beta_p_tol * beta_p_target:
                break
            balanced = False
        prm = params.with_beta0(state["beta0"])
        j, d5, d6, ht, status, converged = self._controlled_solve(vac, ip_A, prm, state["j"], state["d5"], state["d6"], state["r"], z_target,
                                                                  tolerance=tolerance, **common)
        iterations += len(ht)
        at_bound = bool(state["beta0"] <= beta0_bounds[0] + 1e-9 or state["beta0"] >= beta0_bounds[1] - 1e-9)
        balanced = bool(converged and abs(d5) < balance_tol and abs(state["beta_p"] - beta_p_target) < beta_p_tol * beta_p_target)
        return (j, d5, d6, state["history"] + ht, status, converged, state["beta0"], state["beta_p"], state["r"], outer, at_bound,
                iterations, balanced)

    # ---- main solve ---------------------------------------------------- #
    def _controlled_solve(self, vac, ip_A, params, j, d5, d6, r_target, z_target, *, relaxation, max_iterations,
                          tolerance, r_tolerance, z_tolerance, max_dI, started, max_seconds, radial, gain=0.7,
                          max_step=6.0):
        """Picard on the profile with a 2x2 Newton (dI5 symmetric P5, dI6 antisymmetric P6) pinning the current
        centroid to (r_target, z_target) every step. Returns (j, d5, d6, history, status, converged)."""
        history, status, converged = [], "iteration_limit", False
        J = None
        for it in range(max_iterations):
            if wall_time.monotonic() - started > max_seconds:
                status = "time_limit"
                break
            plasma = self.solver.solve(j)

            def new_current(a5, a6):
                jn, _, _ = self.profile_current(vac + a5 * self.T5 + a6 * self.T6 + plasma, params, ip_A)
                return jn, np.array([self.centroid_r(jn, self.RR), self.centroid_z(jn, self.ZZ)])

            j_new, c = new_current(d5, d6)
            target = np.array([r_target, z_target])
            err = c - target
            if True:   # continuous damped control (a dead band makes the position toggle and stalls convergence)
                if J is None or it % 6 == 0:
                    J = np.zeros((2, 2))
                    step = 2.0
                    _, c5 = new_current(d5 + step, d6)
                    _, c6 = new_current(d5, d6 + step)
                    J[:, 0] = (c5 - c) / step
                    J[:, 1] = (c6 - c) / step
                if radial:
                    delta = np.linalg.solve(J, -err)
                else:
                    delta = np.array([0.0, -err[1] / J[1, 1]])
                if not np.isfinite(delta).all():
                    raise ValueError("degenerate_position_response")
                # damped, step-limited control update: the deadbeat step is huge while the plasma is far from
                # its fixed point (bad initial current), so limit the per-iteration control change
                delta = gain * np.clip(delta, -max_step, max_step)
                d5n = float(np.clip(d5 + delta[0], -max_dI, max_dI)) if radial else d5
                d6n = float(np.clip(d6 + delta[1], -max_dI, max_dI))
                j_new, c_new = new_current(d5n, d6n)
                # Broyden update of the response from the (unclipped) step just taken
                dd = np.array([d5n - d5, d6n - d6])
                if np.linalg.norm(dd) > 1e-3 and np.allclose(dd, delta * np.array([radial, 1.0])):
                    J = J + np.outer((c_new - c) - J @ dd, dd) / (dd @ dd)
                d5, d6, c = d5n, d6n, c_new
                err = c - target
            change = float(np.linalg.norm(j_new - j) / max(np.linalg.norm(j), 1e-30))
            history.append(change)
            j = (1.0 - relaxation) * j + relaxation * j_new
            if change < tolerance and abs(err[1]) < 2 * z_tolerance and (not radial or abs(err[0]) < 2 * r_tolerance):
                converged, status = True, "converged"
                break
        return j, d5, d6, history, status, converged

    def solve_frame(self, currents_kA, ip_kA, tf_kA, params: ProfileParams, z_target=0.0, r_target=0.93, *,
                    initial_current=None, relaxation=0.5, max_iterations=80, tolerance=1e-3, r_tolerance=3e-3,
                    z_tolerance=3e-3, max_dI=40.0, max_seconds=90.0, balance_beta0=True, balance_tol=0.5,
                    balance_steps=6, beta0_bounds=(0.02, 0.95), search_tolerance=2e-3, extra_vacuum=None,
                    dn_target=None, dn_steps=4, dn_tolerance=4e-3, beta_p_target=None, beta_p_tol=0.05, beta_p_steps=6,
                    r_init=None):
        """Free-boundary equilibrium of the frame's inputs with prescribed profiles.

        The current centroid is pinned at (r_target, z_target) with a symmetric P5 correction dI5 and an
        antisymmetric P6 correction dI6 (kA-turn) during the iteration. With ``balance_beta0`` an outer secant
        loop adjusts the pressure fraction beta0 of the profile until dI5 = 0 (within ``balance_tol`` kA-turn):
        the REAL vacuum field then holds the plasma at the sampled radius, i.e. beta_p follows from radial force
        balance in the real coil field (Shafranov), and only the small vertical-control dI6 (plus the residual dI5)
        is added to the stored inputs. Returns a dict (see keys below); ``converged`` is the acceptance flag.

        v3 options (default off): ``extra_vacuum`` (65, 65) Wb/rad is ADDED to the coil vacuum field for the solve
        (e.g. unobserved passive-structure flux; NOT reflected in the returned currents). ``dn_target`` requests a
        double-null-oriented equilibrium: after the beta0 balance an outer secant loop on the current-centroid
        target z_target drives the X-point flux asymmetry dn_asym = (psi_x_up - psi_x_low) / (psi_axis - psi_edge)
        to ``dn_target`` (at most ``dn_steps`` warm-started controlled solves, tolerance ``dn_tolerance``).

        v9 options (default off): ``beta_p_target`` switches to the POSITION-FROM-BALANCE mode: the radial position is
        NOT sampled: ``_beta_p_loop`` alternates a beta0 secant (closure beta_p -> ``beta_p_target``, relative tolerance
        ``beta_p_tol``) with a secant on the centroid radius r_target until the radial-control current dI5 vanishes
        (|dI5| < ``balance_tol``), all with warm-started pinned solves; ``r_target``/``r_init`` only seed the search and
        the returned ``r_target`` is the balanced radius. The result carries ``position_mode``, ``beta_p_target``,
        ``beta_p_achieved`` and ``beta0_at_bound`` (beta0 pinned at a bound of ``beta0_bounds`` by either outer loop:
        such equilibria are not force-balanced and can be rejected); ``radial_free`` = balanced AND beta_p matched.
        """
        started = wall_time.monotonic()
        currents = np.asarray(currents_kA, np.float64).copy()
        ip_A = float(ip_kA) * 1000.0
        if not ip_A > 0:
            raise ValueError("positive Ip required (every MAST public frame has Ip > 0)")
        vac = self.vacuum(currents)
        if extra_vacuum is not None:
            vac = vac + np.asarray(extra_vacuum, np.float64)
        position_balance = beta_p_target is not None
        r_seed = r_target if r_init is None else float(r_init)
        j = self.initial_current(ip_A, z_target, Rc=r_seed) if initial_current is None else np.asarray(initial_current, np.float64).copy()
        j *= ip_A / (j.sum() * self.dA)
        common = dict(relaxation=relaxation, max_iterations=max_iterations, r_tolerance=r_tolerance,
                      z_tolerance=z_tolerance, max_dI=max_dI, started=started, max_seconds=max_seconds, radial=True)
        outer, total_iterations = [], 0
        beta0 = float(params.beta0)
        prm = params.with_beta0(beta0)
        at_bound, beta_p_achieved = False, np.nan
        j, d5, d6, history, status, converged = self._controlled_solve(vac, ip_A, prm, j, 0.0, 0.0, r_target, z_target,
                                                                       tolerance=search_tolerance if (balance_beta0 or position_balance) else tolerance, **common)
        total_iterations += len(history)
        outer.append((beta0, d5, converged))
        dn_info = {"dn_mode": 0.0, "dn_target": np.nan, "dn_steps": 0.0, "dn_asym_control": np.nan}
        dn_slope = 1.4          # d(dn_asym)/d(z_target) per metre, rough initial response (pilot: 1.2-1.7)

        def asym_of(jj, a5, a6):
            pl = self.solver.solve(jj)
            ps = vac + a5 * self.T5 + a6 * self.T6 + pl
            surf = plasma_surface(ps, self.R, self.Z, self.envelope, self.lcfs_iter)
            if surf is None:
                return np.nan
            return double_null_metrics(ps, self.R, self.Z, surf, self.envelope)["dn_asym"]

        def dn_secant(j, d5, d6, z_cur, a_cur, z_next, steps):
            """Warm-started controlled solves moving z_target until dn_asym ~ dn_target. Returns state + slope."""
            nonlocal total_iterations, dn_slope
            z_prev, a_prev = z_cur, a_cur
            for _ in range(int(steps)):
                if abs(a_cur - dn_target) < dn_tolerance or abs(z_next - z_cur) < 2e-4:
                    break
                jt, d5t, d6t, ht, st, ct = self._controlled_solve(vac, ip_A, prm, j.copy(), d5, d6, r_target, z_next,
                                                                  tolerance=search_tolerance, **common)
                total_iterations += len(ht)
                dn_info["dn_steps"] += 1
                if not ct:
                    break
                a_new = asym_of(jt, d5t, d6t)
                if not np.isfinite(a_new):
                    break
                j, d5, d6, z_prev, a_prev, z_cur, a_cur = jt, d5t, d6t, z_cur, a_cur, z_next, a_new
                if abs(a_cur - a_prev) < 1e-6:
                    break
                slope = (a_cur - a_prev) / (z_cur - z_prev)
                if slope > 0:
                    dn_slope = slope
                z_next = float(np.clip(z_cur + (dn_target - a_cur) / dn_slope, -0.15, 0.15))
            return j, d5, d6, z_cur, a_cur
        if dn_target is not None and converged:
            # Double-null sampling mode: outer secant on z_target driving the X-point flux asymmetry to dn_target
            # (done BEFORE the beta0 balance, whose radial adjustment barely changes the up-down asymmetry).
            dn_info["dn_mode"], dn_info["dn_target"] = 1.0, float(dn_target)
            a_cur = asym_of(j, d5, d6)
            if np.isfinite(a_cur):
                z_next = float(np.clip(z_target + (dn_target - a_cur) / dn_slope, -0.15, 0.15))
                j, d5, d6, z_target, a_cur = dn_secant(j, d5, d6, z_target, a_cur, z_next, dn_steps)
                dn_info["dn_asym_control"] = float(a_cur)
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
        elif balance_beta0 and converged:
            best = (abs(d5), j.copy(), d5, d6, beta0, history, status)
            b_prev, d_prev = beta0, d5
            # dI5 decreases with beta0 (more pressure -> more outward force -> stronger field needed);
            # first move from a rough response (~ -50 kA-turn per unit beta0), then secant on dI5(beta0)
            b_next = float(np.clip(beta0 + 0.01 * d5, *beta0_bounds))
            for _ in range(balance_steps):
                if best[0] < balance_tol:
                    break
                if abs(b_next - b_prev) < 1e-4:
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
            # tight final solve at the balanced beta0, warm-started
            j, d5, d6, ht, status, converged = self._controlled_solve(vac, ip_A, prm, j, d5, d6, r_target, z_target,
                                                                      tolerance=tolerance, **common)
            total_iterations += len(ht)
            history = history + ht
            balanced = abs(d5) < balance_tol
        if dn_target is not None and converged and np.isfinite(dn_info["dn_asym_control"]):
            # the beta0 balance may have moved the asymmetry: one more (small) secant correction, then a tight solve
            a_cur = asym_of(j, d5, d6)
            if np.isfinite(a_cur) and abs(a_cur - dn_target) > dn_tolerance:
                z_next = float(np.clip(z_target + (dn_target - a_cur) / dn_slope, -0.15, 0.15))
                j, d5, d6, z_target, a_cur = dn_secant(j, d5, d6, z_target, a_cur, z_next, 2)
                j, d5, d6, ht, status, converged = self._controlled_solve(vac, ip_A, prm, j, d5, d6, r_target, z_target,
                                                                          tolerance=tolerance, **common)
                total_iterations += len(ht)
                history = history + ht
                balanced = abs(d5) < balance_tol
            if np.isfinite(a_cur):
                dn_info["dn_asym_control"] = float(a_cur)
        # Final consistent state: j exactly J(psi) with the final controls, official-depth LCFS.
        plasma = self.solver.solve(j)
        psi = vac + d5 * self.T5 + d6 * self.T6 + plasma
        j_final, lam, surface = self.profile_current(psi, prm, ip_A, n_iter=22)
        final_residual = float(np.linalg.norm(j_final - j) / max(np.linalg.norm(j), 1e-30))
        if position_balance:
            beta_p_achieved = self.beta_p_of(psi, prm, lam, surface, ip_A)
            balanced = bool(converged and abs(d5) < balance_tol and abs(beta_p_achieved - beta_p_target) < beta_p_tol * beta_p_target)
        currents[self.i5u] += d5
        currents[self.i5l] += d5
        currents[self.i6u] += d6
        currents[self.i6l] -= d6
        out = {"psi": psi, "j": j, "currents_kA": currents, "dI5_kAt": d5, "dI6_kAt": d6, "lam": lam, "surface": surface,
               "iterations": total_iterations, "residual": final_residual, "history": history, "status": status,
               "converged": bool(converged and final_residual < 3 * tolerance), "z_target": float(z_target),
               "r_target": float(r_target), "radial_free": bool(balanced), "outer": outer,
               "seconds": wall_time.monotonic() - started, "vac": vac, "ip_A": ip_A, "tf_kA": float(tf_kA), "params": prm,
               "params_sampled": params}
        out.update(dn_info)
        out.update({"position_mode": "balance" if position_balance else "pinned", "beta0_at_bound": bool(at_bound),
                    "beta_p_target": float(beta_p_target) if position_balance else np.nan, "beta_p_achieved": float(beta_p_achieved)})
        return out

    # ---- scalars and diagnostics -------------------------------------- #
    def scalars(self, result):
        """q95, betaN, beta_p, pressure profile, official derived scalars and shape diagnostics on the final psi."""
        psi, params, lam, ip_A = result["psi"], result["params"], result["lam"], result["ip_A"]
        C, (iz, ir), axis_val, edge, inside = result["surface"]
        delta = axis_val - edge
        s = np.clip((axis_val - psi) / delta, 0.0, 1.0)
        Ip_int = ShapeIntegral(params.alpha_p, params.gamma_p)
        If_int = ShapeIntegral(params.alpha_f, params.gamma_f)
        pressure = lam * params.beta0 * delta / self.R0 * Ip_int(s) * inside
        if params.edge_amp > 0:      # v9 pedestal pressure (default off)
            pressure = pressure + lam * params.edge_amp * delta / self.R0 * params.edge_pressure_integral(s) * inside
        Fvac = abs(MU0 / (2.0 * np.pi) * TF_TURNS_MAST * result["tf_kA"] * 1000.0)
        if not Fvac > 0:
            raise ValueError("no positive toroidal field")

        def F_squared(level):
            return Fvac ** 2 + 2.0 * MU0 * lam * self.R0 * (1.0 - params.beta0) * delta * If_int(level)

        Fmin = float(np.min(F_squared(np.linspace(0, 1, 1001))))
        if Fmin <= 0:
            raise ValueError("nonpositive_F_squared")
        # q95 on the s = 0.95 surface (5% of the flux from the edge)
        level = edge + 0.05 * delta
        dz, dr = np.gradient(psi, self.Z, self.R)
        bp = np.hypot(dz, dr) / self.RR
        q95 = None
        for curve in find_contours(psi, level):
            if is_closed(curve) and encloses_point(curve, iz, ir):
                physical = grid_to_physical(curve, self.R, self.Z)
                local_bp = _bilinear(bp, self.R, self.Z, physical[:, 0], physical[:, 1])
                if np.any(local_bp <= 0):
                    raise ValueError("degenerate_bp_on_q95_contour")
                integrand = 1.0 / (physical[:, 0] ** 2 * local_bp)
                dl = np.linalg.norm(np.roll(physical, -1, axis=0) - physical, axis=1)
                q95 = float(np.sqrt(F_squared(0.95)) / (2.0 * np.pi) * np.sum(0.5 * (integrand + np.roll(integrand, -1)) * dl))
                break
        if q95 is None:
            raise ValueError("no_closed_q95_contour")
        mean_p = float(np.sum(pressure[inside] * self.RR[inside]) / np.sum(self.RR[inside]))
        Rgeo = 0.5 * (C[:, 0].max() + C[:, 0].min())
        minor = 0.5 * np.ptp(C[:, 0])
        Bt = Fvac / Rgeo
        betaN = float(100.0 * 2.0 * MU0 * mean_p / Bt ** 2 * minor * Bt / (ip_A / 1e6))
        # beta_p with the li(2) normalisation <Bp> = mu0 Ip / lp
        lp = float(np.sum(np.linalg.norm(np.roll(C, -1, axis=0) - C, axis=1)))
        beta_p = float(2.0 * MU0 * mean_p / (MU0 * ip_A / lp) ** 2)
        # official derived scalars on the official LCFS
        C_off = extract_lcfs(psi, self.R, self.Z, "MAST", self.envelope, self.envelope.astype(float), axis_sign=1.0)
        derived = derive_frame(psi, self.R, self.Z, "MAST", self.envelope, self.envelope.astype(float),
                               contour=C_off, axis_sign=1.0)
        envelope_on_C = _bilinear(self.envelope.astype(float), self.R, self.Z, C[:, 0], C[:, 1])
        contact = float(np.mean(envelope_on_C < 0.75))
        grid_touch = bool(((C[:, 0] < self.R[1]) | (C[:, 0] > self.R[-2]) | (C[:, 1] < self.Z[1]) | (C[:, 1] > self.Z[-2])).any())
        return {"q95": q95, "betaN": betaN, "beta_p": beta_p, "pressure_axis_Pa": float(pressure.max()),
                "pressure_mean_Pa": mean_p, "Fvac_Tm": Fvac, "F_squared_min": Fmin, "Bt_geo_T": float(Bt),
                "derived": derived, "lcfs_official": C_off,
                "lcfs_Rmin": float(C[:, 0].min()), "lcfs_Rmax": float(C[:, 0].max()),
                "lcfs_Zmin": float(C[:, 1].min()), "lcfs_Zmax": float(C[:, 1].max()),
                "envelope_contact_fraction": contact, "grid_touch": grid_touch,
                "psi_axis": axis_val, "psi_edge": edge, "psi_ptp": float(psi.max() - psi.min()),
                "pressure": pressure, "s": s, "Ip_integral": Ip_int, "delta": delta}

    def thomson(self, result, scal, chord_R, chord_Z, rng, noise=0.05, ratio=2.0):
        """Synthetic Thomson p_e = p(psi(chord))/ratio with multiplicative noise (mask applied by the caller)."""
        psi = result["psi"]
        params, lam = result["params"], result["lam"]
        finite = np.isfinite(chord_R) & np.isfinite(chord_Z)
        psi_c = np.zeros(len(chord_R))
        psi_c[finite] = _bilinear(psi, self.R, self.Z, chord_R[finite], chord_Z[finite])
        s = np.clip((scal["psi_axis"] - psi_c) / scal["delta"], 0.0, 1.0)
        inside = np.zeros(len(chord_R), bool)
        C = result["surface"][0]
        inside[finite] = _points_in_polygon(chord_R[finite], chord_Z[finite], C)
        p = lam * params.beta0 * scal["delta"] / self.R0 * scal["Ip_integral"](s) * inside
        if params.edge_amp > 0:      # v9 pedestal pressure (default off)
            p = p + lam * params.edge_amp * scal["delta"] / self.R0 * params.edge_pressure_integral(s) * inside
        pe = p / float(ratio) * (1.0 + noise * rng.standard_normal(len(p)))
        return np.maximum(pe, 0.0).astype(np.float32)


# --------------------------------------------------------------------------- #
# Plausibility filter
# --------------------------------------------------------------------------- #
# Tuned on the pilot (results/challenge2/c2_synth_mast_pilot/report.json): the loose pilot rules gave li p05 0.51 /
# p95 1.13, q95 p95 12.5, a beta0-at-lower-bound tail (beta_p < 0.05, 10%) with |dI5| up to 30 kA-turn on 11% of records.
ACCEPT = {"kappa_min": 1.2, "kappa_max": 3.0, "li_min": 0.45, "li_max": 1.15, "q95_min": 2.0, "q95_max": 12.0,
          "beta_p_min": 0.05, "beta_p_max": 1.5, "contact_max": 0.05, "max_dI6": 10.0, "max_dI5": 1.5}


def accept_equilibrium(result, scal, rules=ACCEPT):
    """(ok, reason) with the task's rejection rules; an accepted record may still be envelope-limited (flagged)."""
    if not result["converged"]:
        return False, f"not_converged:{result['status']}"
    if not np.isfinite(result["psi"]).all():
        return False, "nonfinite_psi"
    d = scal["derived"]
    if scal["lcfs_official"] is None or not np.isfinite(d["kappa"]):
        return False, "no_official_lcfs"
    if scal["grid_touch"]:
        return False, "lcfs_touches_grid"
    if d["kappa"] < rules["kappa_min"] or d["kappa"] > rules["kappa_max"]:
        return False, "kappa_out_of_range"
    if not (rules["li_min"] <= d["li"] <= rules["li_max"]):
        return False, "li_out_of_range"
    if not (rules["q95_min"] <= scal["q95"] <= rules["q95_max"]):
        return False, "q95_out_of_range"
    if not (rules["beta_p_min"] <= scal["beta_p"] <= rules["beta_p_max"]):
        return False, "beta_p_out_of_range"
    if abs(result["dI6_kAt"]) > rules["max_dI6"]:
        return False, "vertical_control_effort"
    if abs(result["dI5_kAt"]) > rules["max_dI5"]:
        return False, "radial_balance_not_met"   # the real vacuum field cannot hold this profile at r_target
    if rules.get("reject_beta0_bounds", False):      # v9 --reject-unbalanced (default off)
        if result.get("beta0_at_bound", False):
            return False, "beta0_at_bound"         # force balance / beta_p target only met by pinning beta0 at a bound
        if result.get("position_mode") == "balance" and not result.get("radial_free", False):
            return False, "beta_p_target_not_met"
    cf = float(rules.get("beta_p_consistency", 0.0))   # v9 chord mode (default off): balanced beta_p vs the Thomson proxy
    if cf > 0 and np.isfinite(result.get("beta_p_ref", np.nan)) and result.get("beta_p_ref", 0.0) > 0:
        ratio = scal["beta_p"] / result["beta_p_ref"]
        if not (1.0 / cf <= ratio <= cf):
            return False, "beta_p_inconsistent"
    if rules.get("chord_tol", 0.0) > 0 and result.get("position_mode") == "chord":
        if not (abs(result.get("chord_R_target", np.nan) - result.get("chord_R_syn", np.nan)) <= rules["chord_tol"]):
            return False, "chord_position_not_met"
    iz, ir = result["surface"][1]
    if scal["envelope_contact_fraction"] > rules["contact_max"]:
        return False, "envelope_limited"
    return True, "ok"


# --------------------------------------------------------------------------- #
# Records, shards and stores
# --------------------------------------------------------------------------- #
RECORD_SCALARS = ["ip_kA", "tf_kA", "time_ms", "time_frac", "time_s", "q95", "betaN", "beta_p", "li", "kappa", "volume",
                  "R_axis", "Z_axis", "tri_top", "tri_bot", "psi_ptp", "psi_axis", "psi_edge", "lcfs_Rmin", "lcfs_Rmax",
                  "lcfs_Zmin", "lcfs_Zmax", "envelope_contact_fraction", "residual", "iterations", "lam", "dI6_kAt", "dI5_kAt",
                  "z_target", "r_target", "radial_free", "pressure_axis_Pa", "seconds", "shot_index", "frame_index", "diverted"]


def make_record(result, scal, shot: cd.ShotInputs, shot_index: int, local_frame: int, pe, pe_mask, diverted: bool, extra: dict | None = None):
    d = scal["derived"]
    T = shot.n_frames
    t = float(shot.times_ms[local_frame])
    span = max(float(shot.times_ms[-1] - shot.times_ms[0]), 1e-3)
    return {"psi": result["psi"].astype(np.float32), "currents": result["currents_kA"].astype(np.float32),
            "pe": pe.astype(np.float32), "pe_mask": np.asarray(pe_mask, bool),
            "chord_R": shot.chord_R.astype(np.float32), "chord_Z": shot.chord_Z.astype(np.float32),
            "params": result["params"].as_array().astype(np.float32),
            "ip_kA": float(result["ip_A"] / 1000.0), "tf_kA": float(result["tf_kA"]), "time_ms": t,
            "time_frac": (t - float(shot.times_ms[0])) / span, "time_s": (t - float(shot.times_ms[0])) / 1000.0,
            "q95": scal["q95"], "betaN": scal["betaN"], "beta_p": scal["beta_p"], "li": d["li"], "kappa": d["kappa"],
            "volume": d["volume"], "R_axis": d["R_axis"], "Z_axis": d["Z_axis"], "tri_top": d["tri_top"], "tri_bot": d["tri_bot"],
            "psi_ptp": scal["psi_ptp"], "psi_axis": scal["psi_axis"], "psi_edge": scal["psi_edge"],
            "lcfs_Rmin": scal["lcfs_Rmin"], "lcfs_Rmax": scal["lcfs_Rmax"], "lcfs_Zmin": scal["lcfs_Zmin"], "lcfs_Zmax": scal["lcfs_Zmax"],
            "envelope_contact_fraction": scal["envelope_contact_fraction"], "residual": result["residual"],
            "iterations": result["iterations"], "lam": result["lam"], "dI6_kAt": result["dI6_kAt"], "dI5_kAt": result["dI5_kAt"],
            "z_target": result["z_target"], "r_target": result["r_target"], "radial_free": bool(result["radial_free"]),
            "pressure_axis_Pa": scal["pressure_axis_Pa"], "seconds": result["seconds"],
            "shot_index": int(shot_index), "frame_index": int(shot.frames[local_frame]), "diverted": bool(diverted),
            "shot_path": str(shot.meta.get("path", shot.shot_key)), "n_shot_frames": int(T)} | (dict(extra) if extra else {})


def write_shard(path: Path, records: list[dict], table_names: list[str]):
    K = max(len(r["pe"]) for r in records)
    n = len(records)
    arrays = {"psi": np.stack([r["psi"] for r in records]).astype(np.float32),
              "currents": np.stack([r["currents"] for r in records]).astype(np.float32),
              "params": np.stack([r["params"] for r in records]).astype(np.float32),
              "pe": np.zeros((n, K), np.float32), "pe_mask": np.zeros((n, K), bool),
              "chord_R": np.full((n, K), np.nan, np.float32), "chord_Z": np.full((n, K), np.nan, np.float32),
              "shot_path": np.asarray([r["shot_path"] for r in records]),
              "n_shot_frames": np.asarray([r["n_shot_frames"] for r in records], np.int64),
              "table_names": np.asarray(table_names), "param_names": np.asarray(PARAM_NAMES)}
    for i, r in enumerate(records):
        k = len(r["pe"])
        arrays["pe"][i, :k], arrays["pe_mask"][i, :k] = r["pe"], r["pe_mask"]
        arrays["chord_R"][i, :k], arrays["chord_Z"][i, :k] = r["chord_R"], r["chord_Z"]
    for key in RECORD_SCALARS:
        arrays[key] = np.asarray([r[key] for r in records], np.float64)
    if all("params_sampled" in r for r in records):
        arrays["params_sampled"] = np.stack([r["params_sampled"] for r in records]).astype(np.float32)
    for key in OPTIONAL_RECORD_ARRAYS:          # v3 metadata (analysis only; absent from v1/v2 shards)
        if all(key in r for r in records):
            arrays[key] = np.stack([np.asarray(r[key]) for r in records]).astype(np.float32)
    for key in OPTIONAL_RECORD_SCALARS + V9_RECORD_SCALARS:      # v9 names only when EVERY record carries them
        if all(key in r for r in records):
            arrays[key] = np.asarray([r[key] for r in records], np.float64)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, **arrays)
    tmp.replace(path)
    return n


def read_shard(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def load_synthetic_shots(data_dir: Path, split: str, table_names: list[str], max_frames: int = 0,
                         diverted_only: bool = False, seed: int = 0) -> list[cd.ShotInputs]:
    """Synthetic records of one split as MAST ``ShotInputs`` (one per source shot; labels phi = +psi, q95/betaN)."""
    data_dir = Path(data_dir)
    manifest = json.loads((data_dir / "manifest.json").read_text())
    shards = [s for s in manifest["shards"] if s["split"] == split]
    if not shards:
        raise ValueError(f"{data_dir}: no shards for split {split}")
    parts = [read_shard(data_dir / s["path"]) for s in shards]
    if list(parts[0]["table_names"]) != list(table_names):
        raise ValueError("synthetic shard current order differs from the MAST machine setup")
    keys = [k for k in parts[0] if k not in ("table_names", "param_names") and all(k in p for p in parts)]
    data = {k: np.concatenate([p[k] for p in parts]) for k in keys}
    keep = np.ones(len(data["psi"]), bool)
    if diverted_only:
        keep &= data["diverted"] > 0.5
    if max_frames and keep.sum() > max_frames:
        idx = np.flatnonzero(keep)
        chosen = np.random.default_rng(seed).choice(idx, size=max_frames, replace=False)
        keep = np.zeros_like(keep)
        keep[np.sort(chosen)] = True
    data = {k: v[keep] for k, v in data.items()}
    shots = []
    for shot_index in np.unique(data["shot_index"]):
        sel = np.flatnonzero(data["shot_index"] == shot_index)
        sel = sel[np.argsort(data["frame_index"][sel], kind="stable")]
        n = len(sel)
        shot = cd.ShotInputs("MAST", f"synth_{int(shot_index):04d}", data["frame_index"][sel].astype(np.int64),
                             data["time_ms"][sel].astype(np.float64), data["currents"][sel].astype(np.float32),
                             data["ip_kA"][sel].astype(np.float32), np.ones(n, bool), 1.0, data["tf_kA"][sel].astype(np.float32),
                             data["chord_R"][sel[0]].astype(np.float32), data["chord_Z"][sel[0]].astype(np.float32),
                             data["pe"][sel].astype(np.float32), data["pe_mask"][sel].astype(bool),
                             data["psi"][sel].astype(np.float32),
                             np.stack([data["q95"][sel], data["betaN"][sel]], axis=1).astype(np.float32),
                             {"path": str(data["shot_path"][sel[0]]), "time_frac": data["time_frac"][sel].astype(np.float32),
                              "time_s": data["time_s"][sel].astype(np.float32), "diverted": data["diverted"][sel] > 0.5,
                              "kappa": data["kappa"][sel], "li": data["li"][sel]})
        shots.append(shot)
    return shots


def synthetic_store(shots: list[cd.ShotInputs], K: int | None = None) -> cd.FrameStore:
    """FrameStore with the REAL shot-relative time_frac/time_s of every synthetic frame (not the subset's)."""
    store = cd.FrameStore(shots, K=K, with_labels=True)
    for i, s in enumerate(shots):
        a, b = store.offsets[i], store.offsets[i + 1]
        store.time_frac[a:b] = s.meta["time_frac"]
        store.time_s[a:b] = s.meta["time_s"]
    return store


# --------------------------------------------------------------------------- #
# Official-scorer evaluation on synthetic MAST frames
# --------------------------------------------------------------------------- #
def _reference_task(args):
    psi_gt, R, Z, mask = args
    import local_score as official
    official.MACHINE = "MAST"
    return official.build_reference(psi_gt, R, Z, mask, mask.astype(np.float64))


def _score_task(args):
    gt, ref, pred, R, Z, mask, psi_sign, means = args
    import local_score as official
    official.MACHINE = "MAST"
    return official.score_shot(gt, ref, pred, R, Z, mask, mask.astype(np.float64), psi_sign, means)


def score_synthetic(shots: list[cd.ShotInputs], pred_psi: np.ndarray, pred_scalars: np.ndarray, workers: int = 4,
                    reference_cache: Path | None = None) -> dict:
    """Official composite S on synthetic MAST frames: starter/local_score.py machinery with MACHINE = 'MAST'.

    ``pred_psi`` (n, 65, 65) and ``pred_scalars`` (n, 2) follow the store order of ``synthetic_store(shots)``.
    The reference (LCFS + derived scalars of the synthetic psi) is cached per call site when a path is given.
    """
    from concurrent.futures import ProcessPoolExecutor
    if str(ROOT / "starter") not in sys.path:
        sys.path.insert(0, str(ROOT / "starter"))
    import local_score as official
    from metrics import Accum, finalize_machine
    from common import N_CONS, N_SCALARS, PSI_SIGNS, SCALARS
    R, Z, mask = pf.load_envelope("MAST")
    gts = [{"psi": s.phi.astype(np.float32), "q95": s.scalars[:, 0].astype(np.float64), "betaN": s.scalars[:, 1].astype(np.float64)}
           for s in shots]
    refs = None
    if reference_cache is not None and Path(reference_cache).exists():
        with np.load(reference_cache, allow_pickle=True) as data:
            refs = list(data["refs"])
    if refs is None:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            refs = list(pool.map(_reference_task, [(g["psi"], R, Z, mask) for g in gts]))
        if reference_cache is not None:
            Path(reference_cache).parent.mkdir(parents=True, exist_ok=True)
            np.savez(reference_cache, refs=np.asarray(refs, dtype=object))
    psi_sum = psi_sumsq = psi_n = 0.0
    scal_sum, scal_sumsq, scal_n = np.zeros(N_SCALARS), np.zeros(N_SCALARS), np.zeros(N_SCALARS)
    cons_sum, cons_sumsq, cons_n = np.zeros(N_CONS), np.zeros(N_CONS), np.zeros(N_CONS)
    for g, ref in zip(gts, refs):
        psi = g["psi"].astype(np.float64)
        psi_sum += psi.sum(); psi_sumsq += (psi ** 2).sum(); psi_n += psi.size
        for j, name in enumerate(SCALARS):
            v = g[name][np.isfinite(g[name])]
            scal_sum[j] += v.sum(); scal_sumsq[j] += (v ** 2).sum(); scal_n[j] += v.size
        cons_gt, cmask = ref[1], ref[2]
        for j in range(N_CONS):
            v = cons_gt[cmask[:, j], j]
            cons_sum[j] += v.sum(); cons_sumsq[j] += (v ** 2).sum(); cons_n[j] += v.size
    ref_stats = {"psi_sum": psi_sum, "psi_sumsq": psi_sumsq, "psi_n": psi_n, "scal_sum": scal_sum, "scal_sumsq": scal_sumsq,
                 "scal_n": scal_n, "cons_sum": cons_sum, "cons_sumsq": cons_sumsq, "cons_n": cons_n}
    mean_psi = psi_sum / psi_n
    means = (mean_psi, np.where(scal_n > 0, scal_sum / np.maximum(scal_n, 1), 0.0),
             np.where(cons_n > 0, cons_sum / np.maximum(cons_n, 1), 0.0))
    preds, start = [], 0
    for s in shots:
        n = s.n_frames
        preds.append({"psirz": np.asarray(pred_psi[start:start + n], np.float64), "q95": np.asarray(pred_scalars[start:start + n, 0], np.float64),
                      "betaN": np.asarray(pred_scalars[start:start + n, 1], np.float64)})
        start += n
    totals = {sg: 0.0 for sg in PSI_SIGNS}
    for g, p in zip(gts, preds):
        rr, _, _ = official.psi_residuals(g["psi"], p["psirz"], mean_psi)
        for sg in PSI_SIGNS:
            totals[sg] += rr[sg]
    psi_sign = min(PSI_SIGNS, key=lambda sg: totals[sg])
    acc = Accum()
    acc.psi_sign = psi_sign
    with ProcessPoolExecutor(max_workers=workers) as pool:
        parts = list(pool.map(_score_task, [(g, ref, p, R, Z, mask, psi_sign, means) for g, ref, p in zip(gts, refs, preds)]))
    for part in parts:
        acc.add(part)
    result = finalize_machine(acc, ref_stats)
    result["frames"] = int(sum(s.n_frames for s in shots))
    result["shots"] = len(shots)
    result["machine"] = "MAST(synthetic)"
    return result


# --------------------------------------------------------------------------- #
# v3 additions (all default-off): passive-structure currents, X-point / double-null tools
# --------------------------------------------------------------------------- #
# Approximate axisymmetric conducting structures of MAST as toroidal rectangular elements. PROVENANCE: this
# is an adjustable table written from public, general knowledge of the MAST machine (cylindrical vacuum vessel
# ~4 m diameter / ~4.4 m tall, i.e. wall at R ~ 2.0 m over |Z| < 2.2 m; centre-column casing outside the
# solenoid at R ~ 0.19 m; horizontal vessel end plates; divertor/passive plates at |Z| ~ 1.5-1.8 m, R 0.6-1.4 m).
# It is NOT taken from any FreeGSNKE / MAST-U machine-description file, nor from any label. Element positions,
# sizes and group shares are modelling choices; the sampled currents are random and unobserved by the network.
PASSIVE_GEOMETRY_VERSION = "mast_passive_v1"
PASSIVE_GEOMETRY = [
    # name, R_centre, Z_from, Z_to (or single Z), R_from, R_to (or single R), n_elements, width, height, base share of I_v
    {"name": "outer_wall", "R": 2.03, "Z": (-2.2, 2.2), "n": 89, "width": 0.03, "height": 0.05, "share": 0.35},
    {"name": "centre_column", "R": 0.19, "Z": (-1.6, 1.6), "n": 65, "width": 0.02, "height": 0.05, "share": 0.35},
    {"name": "end_plate_upper", "Z": 2.2, "R": (0.25, 1.95), "n": 18, "width": 0.10, "height": 0.03, "share": 0.075},
    {"name": "end_plate_lower", "Z": -2.2, "R": (0.25, 1.95), "n": 18, "width": 0.10, "height": 0.03, "share": 0.075},
    {"name": "div_plate_upper", "Z": 1.62, "R": (0.6, 1.4), "n": 17, "width": 0.05, "height": 0.03, "share": 0.075},
    {"name": "div_plate_lower", "Z": -1.62, "R": (0.6, 1.4), "n": 17, "width": 0.05, "height": 0.03, "share": 0.075},
]
PASSIVE_GEOMETRIES = {PASSIVE_GEOMETRY_VERSION: PASSIVE_GEOMETRY}


def passive_elements(geometry=None):
    """Flatten the structure table into element arrays (R, Z, width, height, group index, group names)."""
    geometry = PASSIVE_GEOMETRY if geometry is None else geometry
    R, Z, W, H, G, names = [], [], [], [], [], []
    for g, item in enumerate(geometry):
        names.append(item["name"])
        n = int(item["n"])
        if isinstance(item["Z"], (tuple, list)):
            z = np.linspace(item["Z"][0], item["Z"][1], n)
            r = np.full(n, float(item["R"]))
        else:
            r = np.linspace(item["R"][0], item["R"][1], n)
            z = np.full(n, float(item["Z"]))
        R.append(r); Z.append(z); W.append(np.full(n, item["width"])); H.append(np.full(n, item["height"])); G.append(np.full(n, g))
    return (np.concatenate(R), np.concatenate(Z), np.concatenate(W), np.concatenate(H), np.concatenate(G).astype(int), names)


def passive_green_table(R, Z, geometry=None, quadrature_order=3):
    """(NP, nz, nr) Wb/rad per kA of each passive element (uniform current over the element cross-section)."""
    from fusion.mast_physics import ring_green
    eR, eZ, eW, eH, _, _ = passive_elements(geometry)
    RR, ZZ = np.meshgrid(np.asarray(R, float), np.asarray(Z, float))
    nodes, weights = np.polynomial.legendre.leggauss(quadrature_order)
    table = np.zeros((len(eR),) + RR.shape)
    for e in range(len(eR)):
        field = np.zeros_like(RR)
        for i, nr in enumerate(nodes):
            for j, nz in enumerate(nodes):
                a, b = eR[e] + nr * eW[e] / 2, eZ[e] + nz * eH[e] / 2
                try:
                    G = ring_green(RR, ZZ, a, b)
                except ValueError:      # quadrature node coincides with a grid node: nudge by 0.1 mm
                    G = ring_green(RR, ZZ, a + 1e-4, b + 1e-4)
                field += weights[i] * weights[j] / 4 * G
        table[e] = field * 1000.0
    return table


def sample_passive_currents(rng, sigma_total_kA, asym_frac=0.5, share_lognormal_std=0.4, geometry=None):
    """Random low-order passive-current pattern (kA per element).

    total I_v ~ N(0, sigma_total_kA), split between the structure groups with their base shares perturbed by a
    lognormal factor (std ``share_lognormal_std`` in log space) and spread uniformly along each group (n = 0
    term); plus an up/down antisymmetric term I_a ~ N(0, asym_frac * sigma_total_kA) with element weights
    proportional to Z (zero net current). Returns (currents_kA, info dict).
    """
    geometry = PASSIVE_GEOMETRY if geometry is None else geometry
    eR, eZ, eW, eH, eG, names = passive_elements(geometry)
    I_v = float(rng.normal(0.0, sigma_total_kA))
    I_a = float(rng.normal(0.0, asym_frac * sigma_total_kA))
    base = np.asarray([item["share"] for item in geometry], float)
    shares = base * np.exp(rng.normal(0.0, share_lognormal_std, size=len(base)))
    shares /= shares.sum()
    out = np.zeros(len(eR))
    zmax = float(np.abs(eZ).max())
    for g in range(len(geometry)):
        sel = eG == g
        out[sel] += I_v * shares[g] / sel.sum()
    # antisymmetric term: weight ~ Z over ALL elements (structures are up-down symmetric so the net is ~0)
    w = eZ / zmax
    w -= w.mean()
    out += I_a * w / np.abs(w).sum()
    return out, {"passive_total_kA": I_v, "passive_asym_kA": I_a, "passive_shares": shares.tolist()}


def find_xpoints(psi, R, Z, iz_axis, ir_axis, envelope, margin_cells=2):
    """Upper/lower X-point candidates of an axis-maximum psi via a grid saddle search on each half.

    Cells where dpsi/dR and dpsi/dZ both change sign and the finite-difference Hessian has negative determinant
    are saddle candidates; the search is restricted to the envelope dilated by ``margin_cells`` (coils lie
    outside). Each half (Z above / below the axis) returns the saddle with the LARGEST psi (nearest flux to the
    axis = the one that limits or would limit the plasma), refined by a local quadratic fit. Returns dict with
    'up'/'low' -> (R, Z, psi) or None.
    """
    psi = np.asarray(psi, float)
    dz, dr = np.gradient(psi, Z, R)
    dzz, dzr = np.gradient(dz, Z, R)
    _, drr = np.gradient(dr, Z, R)
    det = drr * dzz - dzr ** 2
    from scipy.ndimage import binary_dilation
    region = binary_dilation(envelope, iterations=margin_cells)
    region[[0, 1, -2, -1], :] = False
    region[:, [0, 1, -2, -1]] = False

    def zero_span(a):
        c = np.stack([a[:-1, :-1], a[1:, :-1], a[:-1, 1:], a[1:, 1:]])
        return (c.min(0) <= 0) & (c.max(0) >= 0)
    cand = zero_span(dr) & zero_span(dz) & (det[:-1, :-1] < 0) & region[:-1, :-1]
    out = {"up": None, "low": None}
    axis_val = psi[iz_axis, ir_axis]
    for iz, ir in np.argwhere(cand):
        # node of the cell with the smallest gradient
        block = np.hypot(dr[iz:iz + 2, ir:ir + 2], dz[iz:iz + 2, ir:ir + 2])
        k = np.unravel_index(np.argmin(block), block.shape)
        jz, jr = iz + k[0], ir + k[1]
        if abs(jz - iz_axis) <= 2 and abs(jr - ir_axis) <= 2:
            continue
        # quadratic refinement around (jz, jr)
        g = np.array([dr[jz, jr], dz[jz, jr]])
        Hm = np.array([[drr[jz, jr], dzr[jz, jr]], [dzr[jz, jr], dzz[jz, jr]]])
        if np.linalg.det(Hm) >= 0:
            continue
        step = -np.linalg.solve(Hm, g)
        hR, hZ = R[1] - R[0], Z[1] - Z[0]
        if abs(step[0]) > 1.5 * hR or abs(step[1]) > 1.5 * hZ:
            step = np.clip(step, [-hR, -hZ], [hR, hZ])
        val = float(psi[jz, jr] + g @ step + 0.5 * step @ Hm @ step)
        if val >= axis_val:
            continue
        r, z = float(R[jr] + step[0]), float(Z[jz] + step[1])
        side = "up" if z > Z[iz_axis] else "low"
        if out[side] is None or val > out[side][2]:
            out[side] = (r, z, val)
    return out


def double_null_metrics(psi, R, Z, surface, envelope):
    """X-point flux asymmetry and outboard-midplane separatrix gap of an equilibrium (analysis + control observable).

    dn_asym = (psi_x_up - psi_x_low) / (psi_axis - psi_edge); positive: the upper X-point limits the plasma.
    dr_sep_m: outboard midplane radial distance between the two separatrices (NaN if either X-point missing).
    """
    C, (iz, ir), axis_val, edge, inside = surface
    xp = find_xpoints(psi, R, Z, iz, ir, envelope)
    up, low = xp["up"], xp["low"]
    m = {"xpt_up_R": np.nan, "xpt_up_Z": np.nan, "xpt_up_psi": np.nan, "xpt_low_R": np.nan, "xpt_low_Z": np.nan,
         "xpt_low_psi": np.nan, "dn_asym": np.nan, "dr_sep_m": np.nan, "n_xpoints": int(up is not None) + int(low is not None)}
    if up is not None:
        m["xpt_up_R"], m["xpt_up_Z"], m["xpt_up_psi"] = up
    if low is not None:
        m["xpt_low_R"], m["xpt_low_Z"], m["xpt_low_psi"] = low
    if up is not None and low is not None:
        delta = axis_val - edge
        m["dn_asym"] = float((up[2] - low[2]) / delta)
        # outboard midplane gap: along the axis row, radii where psi crosses the primary and secondary X-point flux
        row = psi[iz, ir:]
        rr = R[ir:]
        gaps = []
        for level in (up[2], low[2]):
            below = np.flatnonzero(row < level)
            if len(below) == 0:
                gaps.append(np.nan)
                continue
            k = below[0]
            if k == 0:
                gaps.append(np.nan)
                continue
            f = (row[k - 1] - level) / max(row[k - 1] - row[k], 1e-30)
            gaps.append(float(rr[k - 1] + f * (rr[k] - rr[k - 1])))
        if np.isfinite(gaps).all():
            m["dr_sep_m"] = float(abs(gaps[0] - gaps[1]))
    return m


def classify_double_null(dn_asym, connected=0.01, near=0.03):
    """0: single-null lower, 1: single-null upper, 2: near double-null, 3: connected double-null, -1: unknown."""
    if dn_asym is None or not np.isfinite(dn_asym):
        return -1
    if abs(dn_asym) < connected:
        return 3
    if abs(dn_asym) < near:
        return 2
    return 1 if dn_asym > 0 else 0


# Optional per-record arrays/scalars written by write_shard when EVERY record of a shard carries them (v3 generator).
OPTIONAL_RECORD_ARRAYS = ["currents_solve", "passive_currents_kA"]
OPTIONAL_RECORD_SCALARS = ["passive_total_kA", "passive_asym_kA", "psi_passive_axis", "psi_passive_boundary_mean",
                           "psi_passive_boundary_absmax", "psi_passive_rms", "pe_ratio", "coil_noise_std", "dn_mode", "dn_target",
                           "dn_asym", "dn_class", "dn_steps", "dr_sep_m", "n_xpoints", "xpt_up_R", "xpt_up_Z", "xpt_up_psi",
                           "xpt_low_R", "xpt_low_Z", "xpt_low_psi", "coil_noise_psi_axis", "coil_noise_psi_rms"]


# --------------------------------------------------------------------------- #
# v9 additions (all default-off): input-derived beta_p proxy, position from balance, edge current
# --------------------------------------------------------------------------- #
# MEMORY (from results/analysis/synth_fidelity_d3d, DIII-D labels): the v1-v3 generator samples the current-centroid
# radius from a currents-independent prior (MAST v1 U(0.76, 0.90) m) that sits ~0.11 m inside the true centroid on
# DIII-D (D_LCFS 0.109 as run vs 0.030 with the right position); the position must therefore come from radial force
# balance in the REAL vacuum field with a physically plausible beta_p, which is estimated here from the frame's own
# Thomson pressure (label-free), not from a prior.
KAPPA_EST_DEFAULT = 1.8         # elongation assumed for the perimeter of the beta_p proxy (MAST/DIII-D typical); the
                                # +-30 % lognormal jitter of the generator covers this and the other proxy approximations


def _smooth_profile(p, size=5):
    from scipy.ndimage import median_filter, uniform_filter1d
    p = np.asarray(p, np.float64)
    if len(p) < size:
        return p.copy()
    return uniform_filter1d(median_filter(p, size=size, mode="nearest"), size=3, mode="nearest")


def _hot_extent(x, p, k, threshold):
    """Contiguous region around index k where p > threshold: (x_lo, x_hi, hit_lo, hit_hi) with linear crossings."""
    lo_hit = hi_hit = False
    x_lo, x_hi = x[0], x[-1]
    for i in range(k, 0, -1):
        if p[i - 1] <= threshold:
            f = (p[i] - threshold) / max(p[i] - p[i - 1], 1e-30)
            x_lo, lo_hit = x[i] - f * (x[i] - x[i - 1]), True
            break
    for i in range(k, len(p) - 1):
        if p[i + 1] <= threshold:
            f = (p[i] - threshold) / max(p[i] - p[i + 1], 1e-30)
            x_hi, hi_hit = x[i] + f * (x[i + 1] - x[i]), True
            break
    return float(x_lo), float(x_hi), lo_hit, hi_hit


def _volume_average(rho, p, n_bins=20):
    """<p> = int_0^1 p(rho) 2 rho drho from scattered (rho, p) samples (area-weighted nested surfaces, flat core fill)."""
    rho, p = np.asarray(rho, float), np.asarray(p, float)
    keep = np.isfinite(rho) & np.isfinite(p) & (rho >= 0) & (rho <= 1)
    rho, p = rho[keep], p[keep]
    if len(rho) < 4:
        raise ValueError("proxy_too_few_points")
    edges = np.linspace(0, 1, n_bins + 1)
    centres = 0.5 * (edges[1:] + edges[:-1])
    idx = np.minimum((rho * n_bins).astype(int), n_bins - 1)
    sums, counts = np.bincount(idx, weights=p, minlength=n_bins), np.bincount(idx, minlength=n_bins)
    filled = counts > 0
    prof = np.interp(centres, centres[filled], sums[filled] / counts[filled])   # flat extrapolation into empty bins
    prof = np.maximum(prof, 0.0)
    return float(np.sum(prof * 2.0 * centres) * (1.0 / n_bins)), centres, prof


def thomson_beta_p_proxy(chord_R, chord_Z, pe, mask, ip_A, layout="midplane", threshold=0.1, kappa_est=KAPPA_EST_DEFAULT,
                         d3d_a_est=0.58, d3d_core_R=1.94, min_points=12, extent="foot"):
    """Label-free electron beta_p estimate of one frame from its Thomson chord (Pa in, ``pe`` = e n_e T_e).

    layout 'midplane' (MAST: horizontal chord at Z = 0 across the whole plasma): the smoothed p_e(R) profile gives the
    axis (max), the hot extent R_in..R_out (p_e > ``threshold`` x max, linear crossings), a_est = (R_out - R_in) / 2,
    rho = |R - R_axis| / (half-width on that side) and <p_e> = int p_e(rho) 2 rho drho (both sides binned).
    layout 'd3d' (DIII-D: tangential chord R 1.68-2.06 at Z ~ -0.05 = outboard midplane core, plus the vertical core
    chord at R = 1.94, Z 0.03-0.8 reaching the edge): p_axis and rho = (R - R_axis) / a_est from the tangential chord,
    (``extent`` 'foot': R_in/R_out are the zero-pressure feet of the edge gradient extrapolated from the 3 x threshold
    and threshold crossings; 'threshold': the crossings themselves).
    the vertical chord mapped by rho(Z) = sqrt(rho_R^2 + (1 - rho_R^2) (Z / Z_hot)^2) with Z_hot the Z where p_e drops
    to ``threshold`` x p_axis and rho_R = (1.94 - R_axis) / a_est; a_est is the machine prior ``d3d_a_est`` (DIII-D
    plasmas fill the vessel, a ~ 0.55-0.62 m; NOT from EFIT), R_axis the tangential maximum. Documented approximation.
    Both: beta_p,e = 2 mu0 <p_e> / (mu0 Ip / l_p)^2 with l_p = 2 pi a_est sqrt((1 + kappa_est^2) / 2).
    Returns dict(beta_p_e, p_mean_e, p_axis_e, a_est, R_in, R_out, R_axis_est, Rgeo_est, lp_est, n_points, extent_ok).
    The caller multiplies by the total/electron pressure ratio and applies the lognormal jitter.
    """
    R, Z, p = (np.asarray(x, np.float64) for x in (chord_R, chord_Z, pe))
    m = np.asarray(mask, bool) & np.isfinite(R) & np.isfinite(Z) & np.isfinite(p)
    ip_A = abs(float(ip_A))
    if ip_A <= 0:
        raise ValueError("proxy_needs_ip")
    if layout == "midplane":
        sel = m & (np.abs(Z) < 0.05)
        if sel.sum() < min_points:
            raise ValueError("proxy_too_few_points")
        order = np.argsort(R[sel])
        r, pp = R[sel][order], _smooth_profile(np.maximum(p[sel][order], 0.0))
        k = int(np.argmax(pp))
        p_axis = float(pp[k])
        if not p_axis > 0:
            raise ValueError("proxy_zero_pressure")
        R_in, R_out, lo_hit, hi_hit = _hot_extent(r, pp, k, threshold * p_axis)
        if extent == "foot":
            # the LCFS lies OUTSIDE the threshold crossing (p_e(LCFS) ~ 0): extrapolate the edge gradient between the
            # 3 x threshold and threshold crossings linearly to zero pressure on each side (pilot: the plain 10 %
            # crossing sits ~9 cm inside the synthetic LCFS on MAST, i.e. a 17 % low a_est -> 30 % low beta_p)
            R_in3, R_out3, _, _ = _hot_extent(r, pp, k, 3.0 * threshold * p_axis)
            f = threshold / (3.0 * threshold - threshold)
            R_out = R_out + f * max(R_out - R_out3, 0.0)
            R_in = R_in - f * max(R_in3 - R_in, 0.0)
        # axis estimate: pressure-weighted centre of the contiguous core (p > 0.5 max around the maximum), robust to
        # single-chord spikes of the MAST data; the peak value itself is the smoothed maximum
        lo_c, hi_c, _, _ = _hot_extent(r, pp, k, 0.5 * p_axis)
        core = (r >= lo_c) & (r <= hi_c)
        R_axis = float(np.sum(r[core] * pp[core]) / max(np.sum(pp[core]), 1e-30)) if core.sum() >= 2 else float(r[k])
        R_axis = float(np.clip(R_axis, R_in + 1e-3, R_out - 1e-3))
        a_in, a_out = max(R_axis - R_in, 1e-3), max(R_out - R_axis, 1e-3)
        rho = np.where(r >= R_axis, (r - R_axis) / a_out, (R_axis - r) / a_in)
        p_mean, _, _ = _volume_average(rho, pp)
        a_est = 0.5 * (R_out - R_in)
        extent_ok = bool(lo_hit and hi_hit)
        n_points = int(sel.sum())
    elif layout == "d3d":
        tan = m & (np.abs(Z) < 0.15) & (R > 1.5)
        ver = m & (np.abs(R - d3d_core_R) < 0.03) & (Z > 0.0)
        if tan.sum() < 5 or ver.sum() < min_points:
            raise ValueError("proxy_too_few_points")
        ot = np.argsort(R[tan])
        rt, pt = R[tan][ot], _smooth_profile(np.maximum(p[tan][ot], 0.0), size=3)
        k = int(np.argmax(pt))
        p_axis, R_axis = float(pt[k]), float(rt[k])
        if not p_axis > 0:
            raise ValueError("proxy_zero_pressure")
        a_est = float(d3d_a_est)
        rho_t = (rt - R_axis) / a_est
        ov = np.argsort(Z[ver])
        zv, pv = Z[ver][ov], _smooth_profile(np.maximum(p[ver][ov], 0.0))
        kz = int(np.argmax(pv))
        _, Z_hot, _, hi_hit = _hot_extent(zv, pv, kz, threshold * p_axis)
        rho_R = float(np.clip((d3d_core_R - R_axis) / a_est, 0.0, 0.95))
        rho_v = np.sqrt(rho_R ** 2 + (1.0 - rho_R ** 2) * (zv / max(Z_hot, 1e-3)) ** 2)
        p_mean, _, _ = _volume_average(np.concatenate([rho_t, rho_v]), np.concatenate([pt, pv]))
        R_in, R_out = R_axis - a_est, R_axis + a_est
        extent_ok = bool(hi_hit)
        n_points = int(tan.sum() + ver.sum())
    else:
        raise ValueError(f"unknown proxy layout {layout}")
    lp = 2.0 * np.pi * a_est * np.sqrt(0.5 * (1.0 + kappa_est ** 2))
    bp = MU0 * ip_A / lp
    beta_p_e = float(2.0 * MU0 * p_mean / bp ** 2)
    return {"beta_p_e": beta_p_e, "p_mean_e": float(p_mean), "p_axis_e": p_axis, "a_est": float(a_est), "R_in": float(R_in),
            "R_out": float(R_out), "R_axis_est": R_axis, "Rgeo_est": float(0.5 * (R_in + R_out)), "lp_est": float(lp),
            "n_points": n_points, "extent_ok": extent_ok, "layout": layout, "extent": extent}


def sample_beta_p_target(proxy, rng, ratio_range=(1.5, 2.5), jitter_sigma=0.3, bounds=(0.03, 2.0)):
    """beta_p target = beta_p,e x ratio (total/electron pressure, U(ratio_range)) x lognormal(jitter_sigma), clipped."""
    ratio = float(rng.uniform(*ratio_range)) if ratio_range[1] > ratio_range[0] else float(ratio_range[0])
    jitter = float(np.exp(jitter_sigma * rng.standard_normal())) if jitter_sigma > 0 else 1.0
    raw = proxy["beta_p_e"] * ratio * jitter
    return float(np.clip(raw, *bounds)), {"pe_ratio_proxy": ratio, "beta_p_jitter": jitter, "beta_p_proxy_raw": float(raw),
                                         "beta_p_clipped": bool(raw < bounds[0] or raw > bounds[1])}


def sample_edge_params(rng, amp_range=(0.0, 0.3), w_range=(0.03, 0.12)):
    return float(rng.uniform(*amp_range)), float(rng.uniform(*w_range))


def edge_current_ratio(j, psi, surface, s_edge=0.95):
    """(mean j on s > s_edge inside the LCFS) / j at the axis cell, plus the current fraction on s > 0.9 (label metric)."""
    C, (iz, ir), axis_val, edge, inside = surface
    s = np.clip((axis_val - psi) / (axis_val - edge), 0.0, 1.0)
    shell = inside & (s > s_edge)
    j_axis = float(j[iz, ir])
    j_edge = float(j[shell].mean()) if shell.any() else 0.0
    outer = inside & (s > 0.9)
    total = float(j[inside].sum())
    return {"j_edge_over_axis": j_edge / j_axis if j_axis > 0 else np.nan,
            "edge_I_frac_s_gt_0.9": float(j[outer].sum() / total) if total > 0 else np.nan}


# v9 record scalars (written only when EVERY record of a shard carries them; v1-v3 shards are unaffected)
V9_RECORD_SCALARS = ["position_mode_balance", "beta_p_target", "beta_p_achieved", "beta_p_proxy_e", "pe_ratio_proxy",
                     "beta_p_jitter", "proxy_a_est", "proxy_Rgeo_est", "proxy_R_axis_est", "proxy_extent_ok", "beta0_at_bound",
                     "edge_amp", "edge_w", "j_edge_over_axis", "edge_I_frac_s_gt_0.9", "centroid_R", "centroid_Z",
                     # chord-position mode (iteration 3): the SEED radius, the chord observable target / achieved value,
                     # the number of position-correction re-solves and the proxy beta_p reference of the consistency rule
                     "position_mode_chord", "r_seed", "chord_R_target", "chord_R_syn", "chord_steps", "beta_p_ref"]
# NOTE: OPTIONAL_RECORD_SCALARS is NOT extended (v1-v3 manifests list it verbatim); write_shard looks at both lists.


# --------------------------------------------------------------------------- #
# v9 iteration 3: position from the frame's own midplane Thomson chord (label-free), pressure from force balance
# --------------------------------------------------------------------------- #
# MEMORY (results/analysis/synth_fidelity_d3d_v9/report_iter{1,2}.json, DIII-D labels, 300 frames): the position-from-
# balance mode removes the BIAS of the r_target prior (centroid error median -0.013 m vs -0.106 m) but not the scatter
# (|error| median 0.09 m even with the label beta_p): with the real coil field the balanced radius is set by the
# sampled profile (li, shape) as much as by beta_p, the plasma comes out under-elongated (kappa -0.16) and only 15 %
# of the frames pass the acceptance rules (D_LCFS 0.053 on those, 0.110 on all converged). On MAST the Thomson chord
# is HORIZONTAL at Z = 0 across the whole plasma (146 points, R 0.23-1.52 m, 1 cm spacing), so the plasma position is
# an INPUT observable: the estimator of ``thomson_beta_p_proxy`` (layout 'midplane') gives the pressure core centre
# R_axis_est and the LCFS midplane centre Rgeo_est. ``solve_frame_chord`` pins the centroid (v1-v3 pinned solve with
# beta0 balanced so that the REAL vacuum field holds the plasma, |dI5| < tol) and corrects the centroid target until
# the SAME estimator applied to the equilibrium's own noise-free chord pressure reproduces the measured value (fixed
# point, at most ``steps`` warm-started re-solves): no label, no prior on the radius, no dI5 fudge. The beta0-balanced
# beta_p is then checked against the proxy (``beta_p_consistency`` rule) so that the pressure stays plausible.


def chord_pressure(solver, result, chord_R, chord_Z):
    """Noise-free pressure p(psi(chord)) (Pa) of a solved equilibrium on the given chord points (mask = finite points)."""
    psi, params, lam = result["psi"], result["params"], result["lam"]
    C, _, axis_val, edge, _ = result["surface"]
    R, Z = np.asarray(chord_R, np.float64), np.asarray(chord_Z, np.float64)
    finite = np.isfinite(R) & np.isfinite(Z)
    delta = axis_val - edge
    psi_c = np.zeros(len(R))
    psi_c[finite] = _bilinear(psi, solver.R, solver.Z, R[finite], Z[finite])
    s = np.clip((axis_val - psi_c) / delta, 0.0, 1.0)
    inside = np.zeros(len(R), bool)
    inside[finite] = _points_in_polygon(R[finite], Z[finite], C)
    return solver.pressure_field(s, params, lam, delta, inside), finite


def chord_position_observable(solver, result, chord_R, chord_Z, match="axis", threshold=0.1):
    """The midplane estimator of ``thomson_beta_p_proxy`` applied to the equilibrium's OWN chord pressure: returns
    (R_axis_est or Rgeo_est of the synthetic chord, full estimate dict). Raises ValueError like the proxy."""
    p, finite = chord_pressure(solver, result, chord_R, chord_Z)
    est = thomson_beta_p_proxy(chord_R, chord_Z, p, finite, result["ip_A"], layout="midplane", threshold=threshold)
    return float(est["R_axis_est"] if match == "axis" else est["Rgeo_est"]), est


def solve_frame_chord(solver, target_R, chord_R, chord_Z, solve_args, *, match="axis", steps=2, tol=5e-3, r_bounds=(0.55, 1.15),
                      threshold=0.1, gain=1.0, r_target, initial_current=None, **solve_kw):
    """Chord-position mode: pinned solve at ``r_target`` (seed), then r_target += gain x (target_R - observable) with the
    observable ('axis' = pressure core centre, 'rgeo' = LCFS midplane centre) measured on the equilibrium's own chord,
    re-solved warm-started at most ``steps`` times (stop when |target_R - observable| < ``tol`` m). ``solve_args`` are
    the positional arguments of ``solver.solve_frame`` (currents, Ip, TF, params); every solve keeps the beta0 balance
    of the pinned mode. Returns (result, scal_or_None): result carries r_seed, chord_R_target, chord_R_syn, chord_steps
    and position_mode 'chord'; scal is ``solver.scalars(result)`` of the returned state when it exists."""
    r = float(r_target)
    info = {"r_seed": r, "chord_R_target": float(target_R), "chord_R_syn": np.nan, "chord_steps": 0, "position_mode": "chord"}
    j0 = initial_current
    for k in range(int(steps) + 1):
        res = solver.solve_frame(*solve_args, r_target=r, initial_current=j0, **solve_kw)
        res.update(info)
        res["chord_steps"] = k
        if not res["converged"] or res.get("surface") is None:
            return res, None
        try:
            r_syn, _ = chord_position_observable(solver, res, chord_R, chord_Z, match=match, threshold=threshold)
        except ValueError as exc:
            res["status"] = f"chord_observable:{exc}"
            res["converged"] = False
            return res, None
        res["chord_R_syn"] = r_syn
        diff = float(target_R) - r_syn
        if abs(diff) < tol or k == int(steps):
            return res, solver.scalars(res)
        r = float(np.clip(r + gain * diff, *r_bounds))
        j0 = res["j"]
    raise RuntimeError("unreachable")
