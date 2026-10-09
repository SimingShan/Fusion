"""Input-only MAST magnetic reconstruction prototype, with explicit closure assumptions.

The coil field and free-space boundary Green function are physical calculations.
The plasma profile is an unvalidated closure, not an inferred EFIT reconstruction.
No MAST labels (including the starter demo labels) are read by this module.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import sys
import time as wall_time

import numpy as np
from scipy import sparse
from scipy.ndimage import median_filter
from scipy.sparse.linalg import splu
from scipy.special import ellipe, ellipk
from scipy.optimize import anderson, newton_krylov, NoConvergence

from fusion.data import arrow_to_numpy, interpolate_signal
from starter.data_fixes import align_mast_thomson_core

MU0 = 4.0e-7 * np.pi
E_CHARGE = 1.602176634e-19
ROOT = Path(__file__).resolve().parents[1]
SCORER = ROOT / "starter" / "fusion_scoring"
if str(SCORER) not in sys.path:
    sys.path.insert(0, str(SCORER))
from contour import encloses_point, find_contours, grid_to_physical, is_closed
from derive import _bilinear, _points_in_polygon, derive_frame
from lcfs import extract_lcfs
from o_point import find_o_point


def read_mast_inputs(path: Path) -> dict:
    """Read an explicit input allowlist, even if the supplied file contains targets."""
    import pyarrow.parquet as pq
    allowed = {
        "source", "efit_times", "efit_grid_R", "efit_grid_Z", "magnetics_time",
        "magnetics_plasma_current", "magnetics_sol_current", "magnetics_tf_current",
        "magnetics_efps_current", "coil_name", "coil_input_column", "coil_R", "coil_Z",
        "coil_width", "coil_height", "coil_angle1", "coil_angle2",
        "thomson_chord_name", "thomson_chord_R", "thomson_chord_Z",
        "thomson_core_times", "thomson_core_R", "thomson_core_Te", "thomson_core_ne",
        "thomson_edge_times", "thomson_edge_spatial", "thomson_edge_Te", "thomson_edge_ne",
    } | {f"magnetics_p{n}{side}_current" for n in range(2, 7) for side in "lu"}
    names = pq.ParquetFile(path).schema_arrow.names
    columns = sorted(allowed.intersection(names))
    table = pq.read_table(path, columns=columns)
    if len(table) != 1:
        raise ValueError("Expected exactly one shot per parquet")
    row = {name: arrow_to_numpy(table[name][0]) for name in columns}
    if row.get("source") != "MAST":
        raise ValueError("This prototype accepts MAST inputs only")
    return row


def ring_green(R, Z, source_R, source_Z):
    """Flux per radian [Wb/rad/A] from an axisymmetric circular filament.

    Psi=R*A_phi, with psi→0 at R→0 and spatial infinity. This fixes a physical
    free-space gauge; agreement with the target EFIT gauge is not established.
    Broadcast arguments. Exact filament coincidence is singular and rejected.
    """
    R, Z, a, b = np.broadcast_arrays(R, Z, source_R, source_Z)
    if np.any(R < 0) or np.any(a <= 0):
        raise ValueError("Nonnegative field radius and positive source radius required")
    m = 4.0 * R * a / ((R + a) ** 2 + (Z - b) ** 2)
    if np.any(m >= 1):
        raise ValueError("Filament coincidence: integrate the finite conductor cross-section")
    # The small-m series avoids cancellation of the two elliptic terms near axis.
    out = np.empty_like(m, dtype=np.float64)
    small = m < 1.0e-5
    out[small] = MU0 * np.sqrt(R[small] * a[small]) * m[small] ** 1.5 / 32.0 * (
        1.0 + 0.75 * m[small] + 75.0 / 128.0 * m[small] ** 2)
    k = np.sqrt(m[~small])
    out[~small] = MU0 / (2.0 * np.pi) * np.sqrt(R[~small] * a[~small]) * (
        (2.0 - m[~small]) * ellipk(m[~small]) - 2.0 * ellipe(m[~small])) / k
    return out


@dataclass
class CoilResponse:
    columns: list[str]
    psi_per_kilo_unit: np.ndarray
    conductor_counts: dict
    geometry_sha256: str

    def at(self, row: dict, times: np.ndarray):
        currents, observed = [], []
        for key in self.columns:
            value, mask = interpolate_signal(row["magnetics_time"], row[key], times)
            currents.append(value)
            observed.append(mask)
        currents = np.stack(currents, axis=1).astype(np.float64)
        return np.einsum("tc,czr->tzr", currents, self.psi_per_kilo_unit), np.stack(observed, axis=1)


def coil_response(row: dict, quadrature_order: int = 2) -> CoilResponse:
    """Integrate MAST rectangles, correctly separating kA-turn and kA feeds."""
    join = np.asarray(row["coil_input_column"])
    columns = sorted(set(join.tolist()))
    geometry = np.stack([np.asarray(row[k], dtype=np.float64) for k in
                         ("coil_R", "coil_Z", "coil_width", "coil_height")])
    if np.any(np.asarray(row["coil_angle1"]) != 0) or np.any(np.asarray(row["coil_angle2"]) != 0):
        raise ValueError("Only the published unskewed MAST conductors are supported")
    RR, ZZ = np.meshgrid(row["efit_grid_R"], row["efit_grid_Z"])
    nodes, weights = np.polynomial.legendre.leggauss(quadrature_order)
    response = []
    counts = {}
    for key in columns:
        mask = join == key
        counts[key] = int(mask.sum())
        if key == "magnetics_sol_current":
            per_element = 1000.0  # kA feed, same current in each of the 656 turns
        elif key.startswith("magnetics_p") and key.endswith("_current"):
            per_element = 1000.0 / counts[key]  # already total kA-turn
        else:
            raise ValueError(f"Unknown conductor current convention: {key}")
        a, b, width, height = geometry[:, mask]
        field = np.zeros_like(RR)
        for i, nr in enumerate(nodes):
            for j, nz in enumerate(nodes):
                G = ring_green(RR[..., None], ZZ[..., None],
                               a + nr * width / 2, b + nz * height / 2)
                field += weights[i] * weights[j] / 4 * G.sum(axis=-1)
        response.append(field * per_element)
    fingerprint = hashlib.sha256(geometry.tobytes() + "|".join(join).encode()).hexdigest()
    return CoilResponse(columns, np.stack(response), counts, fingerprint)


class FreeBoundaryGS:
    """Second-order Delta* solve with free-space Green boundary for plasma current.

    Coils remain analytic and outside this finite-difference solve. Plasma sources
    are integrated as rectangular cells using 2x2 Gaussian quadrature on the edge.
    """
    def __init__(self, R, Z):
        self.R, self.Z = np.asarray(R, float), np.asarray(Z, float)
        self.RR, self.ZZ = np.meshgrid(self.R, self.Z)
        self.shape = self.RR.shape
        self.dR, self.dZ = float(np.mean(np.diff(R))), float(np.mean(np.diff(Z)))
        if not np.allclose(np.diff(R), self.dR) or not np.allclose(np.diff(Z), self.dZ):
            raise ValueError("Uniform physical grids required")
        nz, nr = self.shape
        A = sparse.eye(nz * nr, format="lil")
        for iz in range(1, nz - 1):
            for ir in range(1, nr - 1):
                k = iz * nr + ir
                A[k, k] = -2 / self.dR**2 - 2 / self.dZ**2
                A[k, k - 1] = 1 / self.dR**2 + 1 / (2 * R[ir] * self.dR)
                A[k, k + 1] = 1 / self.dR**2 - 1 / (2 * R[ir] * self.dR)
                A[k, k - nr] = A[k, k + nr] = 1 / self.dZ**2
        self.matrix = A.tocsr()
        self.solve_linear = splu(A.tocsc()).solve
        boundary = np.zeros(self.shape, bool)
        boundary[[0, -1], :] = True
        boundary[:, [0, -1]] = True
        self.boundary = boundary.ravel()
        self.source_indices = np.flatnonzero(~self.boundary)
        rsource = self.RR.ravel()[self.source_indices]
        zsource = self.ZZ.ravel()[self.source_indices]
        reval = self.RR.ravel()[self.boundary, None]
        zeval = self.ZZ.ravel()[self.boundary, None]
        self.boundary_green = np.zeros((self.boundary.sum(), len(rsource)))
        for u in (-1 / np.sqrt(3), 1 / np.sqrt(3)):
            for v in (-1 / np.sqrt(3), 1 / np.sqrt(3)):
                self.boundary_green += 0.25 * ring_green(
                    reval, zeval, rsource + u * self.dR / 2, zsource + v * self.dZ / 2)
        self.boundary_green *= self.dR * self.dZ

    def solve(self, jtor):
        jtor = np.asarray(jtor, float)
        if jtor.shape != self.shape or not np.isfinite(jtor).all():
            raise ValueError("Finite native-grid current density required")
        if np.any(jtor.ravel()[self.boundary] != 0):
            raise ValueError("Plasma current must vanish at the computational boundary")
        rhs = (-MU0 * self.RR * jtor).ravel()
        rhs[self.boundary] = self.boundary_green @ jtor.ravel()[self.source_indices]
        return self.solve_linear(rhs).reshape(self.shape)

    def residual(self, psi_plasma, jtor):
        residual = self.matrix @ psi_plasma.ravel() + MU0 * self.RR.ravel() * jtor.ravel()
        interior = ~self.boundary
        scale = np.linalg.norm((MU0 * self.RR * jtor).ravel()[interior])
        return float(np.linalg.norm(residual[interior]) / max(scale, 1e-30))


def thomson_at(row: dict, time: float):
    R, te, ne = align_mast_thomson_core(row)
    values, coverage = [], []
    for data in (te, ne):
        columns = [interpolate_signal(row["thomson_core_times"], data[:, j], np.array([time]))
                   for j in range(len(R))]
        values.append(np.array([x[0][0] for x in columns], dtype=float))
        coverage.append(np.array([x[1][0] for x in columns], dtype=bool))
    valid = (values[0] > 0) & (values[1] > 0)
    observed = valid & coverage[0] & coverage[1]
    pressure = np.where(valid, E_CHARGE * values[0] * values[1], 0.0)
    pressure = median_filter(pressure, size=5, mode="nearest")
    return R, pressure, valid, observed


def _surface(psi, solver, envelope):
    C = extract_lcfs(psi, solver.R, solver.Z, "MAST", envelope, envelope.astype(float),
                     axis_sign=1.0, n_iter=22, n_points=256)
    axis = find_o_point(psi, solver.R, solver.Z, envelope)
    if C is None or not axis[-1]:
        return None
    edge = float(np.mean(_bilinear(psi, solver.R, solver.Z, C[:, 0], C[:, 1])))
    delta = float(psi[axis[2], axis[3]] - edge)
    inside = _points_in_polygon(solver.RR.ravel(), solver.ZZ.ravel(), C).reshape(solver.shape)
    if delta <= 1e-8 or inside.sum() < 12:
        return None
    return C, axis, edge, delta, inside


def _profile_current(psi, solver, envelope, ip, paxis, exponent):
    surface = _surface(psi, solver, envelope)
    if surface is None:
        raise ValueError("no_closed_plasma_surface")
    _, _, edge, delta, inside = surface
    s = np.clip((psi-edge)/delta, 0, 1)
    profile = s**exponent * inside
    jp = solver.RR*paxis*(exponent+1)/delta*profile
    integral = solver.dR*solver.dZ
    A = (ip-jp.sum()*integral)/((profile/solver.RR).sum()*integral)
    current = jp+A/solver.RR*profile
    if np.min(current) < -1e-3*np.max(current):
        raise ValueError("closure_requires_countercurrent")
    return current


class _EvaluationLimit(RuntimeError):
    pass


class _ClosureConverged(RuntimeError):
    pass


def project_current_simplex(current, active, solver, ip):
    """Euclidean projection onto nonnegative currents with measured integral Ip.

    The shift to zero maximum keeps the simplex threshold stable for large
    common offsets. This acts on solver trial currents, never on flux pixels.
    """
    current = np.asarray(current, dtype=float)
    if (current.shape != solver.shape or active.shape != solver.shape or not active.any()
            or np.any(active.ravel()[solver.boundary]) or not np.isfinite(current).all()
            or not np.isfinite(ip) or ip <= 0 or solver.dR*solver.dZ <= 0):
        raise ValueError("Invalid physical current projection")
    count = int(active.sum())
    scale = ip/(count*solver.dR*solver.dZ)
    values = current[active]/scale
    values = values-values.max()
    sorted_values = np.sort(values)[::-1]
    thresholds = (np.cumsum(sorted_values)-count)/np.arange(1, count+1)
    feasible = np.flatnonzero(sorted_values > thresholds)
    if not len(feasible):
        raise ValueError("No finite current projection")
    weights = np.maximum(values-thresholds[feasible[-1]], 0.)
    if not np.isfinite(weights).all() or weights.sum() <= 0:
        raise ValueError("Nonfinite physical current projection")
    projected = np.zeros(solver.shape)
    projected[active] = weights*(ip/(weights.sum()*solver.dR*solver.dZ))
    return projected


def solve_profile_current(vacuum, initial_current, solver, envelope, ip, paxis, exponent,
                          *, method="picard", iterations=200, relaxation=0.2,
                          max_evaluations=400, max_seconds=15.0, tolerance=2e-3,
                          profile_function=None, positive_trials=False):
    """Bounded fixed-coil source residual solve, with no free field/gauge parameters.

    Anderson and Newton/Krylov solve J(psi_vac+G j)-j=0 on the public envelope.
    Trial directions may leave the admissible domain; a failed trial is reported.
    The best evaluated *physical residual* state is retained, never an arbitrary
    successful SciPy flag. Output is renormalized to Ip and checked again.
    """
    started = wall_time.monotonic()
    active = envelope.copy()
    active.ravel()[solver.boundary] = False
    current_scale = ip/(active.sum()*solver.dR*solver.dZ)
    initial_current = np.asarray(initial_current, dtype=float)
    evaluations, invalid_evaluations = 0, 0
    best_current = initial_current.copy()
    best_error = np.inf
    history = []
    reason = "iteration_limit"

    def evaluate(current, *, enforce_budget=True):
        nonlocal evaluations, invalid_evaluations, best_current, best_error
        if enforce_budget and (evaluations >= max_evaluations or wall_time.monotonic()-started >= max_seconds):
            raise _EvaluationLimit("evaluation_or_runtime_limit")
        evaluations += 1
        if not np.isfinite(current).all():
            raise ValueError("nonfinite_current_trial")
        if positive_trials:
            current = project_current_simplex(current, active, solver, ip)
        plasma = solver.solve(current)
        try:
            target = (profile_function(vacuum+plasma) if profile_function is not None else
                      _profile_current(vacuum+plasma, solver, envelope, ip, paxis, exponent))
        except ValueError:
            invalid_evaluations += 1
            raise
        error = float(np.linalg.norm(target-current)/max(np.linalg.norm(current), 1e-30))
        if error < best_error:
            best_error, best_current = error, current.copy()
        return target, error

    def residual(x):
        current = np.zeros(solver.shape)
        current[active] = x*current_scale
        target, error = evaluate(current)
        if error < tolerance:
            raise _ClosureConverged()
        # Keep the optimization variable unprojected in this residual. A root
        # is x=J(P(x)), hence it is itself physically admissible. Convergence
        # and retained states are independently assessed at the physical P(x).
        return (target[active]-current[active])/current_scale

    def callback(x, f):
        history.append(float(np.linalg.norm(f)/max(np.linalg.norm(x),1e-30)))

    current = initial_current.copy()
    try:
        # Shared modest Picard initialization provides an actual plasma surface.
        # Continuation callers can give a prior solved source instead of a TS seed.
        seed_steps = iterations if method == "picard" else min(20, iterations)
        for _ in range(seed_steps):
            target, error = evaluate(current)
            history.append(error)
            if error < tolerance:
                raise _ClosureConverged()
            current = (1-relaxation)*current+relaxation*target
        if method != "picard":
            # A vertical instability may already increase residual during seeding.
            # The best source has the same physical inputs and exact total current.
            x0 = best_current[active]/current_scale
            if method == "krylov":
                x = newton_krylov(residual,x0,maxiter=iterations,rdiff=1e-3,
                                  method="lgmres",inner_maxiter=15,outer_k=5,
                                  f_tol=1e-6,line_search="armijo",callback=callback)
            elif method == "anderson":
                x = anderson(residual,x0,maxiter=iterations,M=6,w0=0.01,alpha=relaxation,
                             f_tol=1e-6,line_search="armijo",callback=callback)
            else:
                raise ValueError(f"Unknown nonlinear method: {method}")
            current = np.zeros(solver.shape)
            current[active] = x*current_scale
            evaluate(current)
    except _ClosureConverged:
        reason = "converged"
    except _EvaluationLimit as e:
        reason = str(e)
    except NoConvergence:
        reason = "iteration_limit"
    except (ValueError, np.linalg.LinAlgError) as e:
        reason = "nonlinear_trial_failure:"+str(e)
    if method != "picard":
        current = best_current
    if positive_trials:
        current = project_current_simplex(current, active, solver, ip)
    integral = current.sum()*solver.dR*solver.dZ
    if integral > 0:
        current = current*(ip/integral)
    final_error = None
    projection_applied = False
    try:
        final_target, final_error = evaluate(current,enforce_budget=False)
        # Accelerated linear combinations can create small negative numerical
        # tails. A single genuine closure map (not pixel clipping) removes them.
        # Accept this projection only if its recomputed residual still converges.
        negative_mass = -np.minimum(current,0).sum()/max(np.abs(current).sum(),1e-30)
        if method != "picard" and negative_mass >= 1e-4 and final_error < tolerance:
            _, projected_error = evaluate(final_target,enforce_budget=False)
            if projected_error < tolerance:
                current, final_error = final_target, projected_error
                projection_applied = True
    except ValueError as e:
        reason = "final_closure_failure:"+str(e)
    converged = final_error is not None and final_error < tolerance
    if converged:
        reason = "converged"
    return current, {"nonlinear_profile_relative_residual":final_error,
        "nonlinear_residual_history":history,"converged":converged,"status":reason,
        "solver_method":method,"function_evaluations":evaluations,"invalid_evaluations":invalid_evaluations,
        "best_evaluated_profile_residual":float(best_error) if np.isfinite(best_error) else None,
        "solver_elapsed_seconds":wall_time.monotonic()-started,
        "evaluation_limit":max_evaluations,"runtime_limit_seconds":max_seconds,
        "convergence_tolerance":tolerance,"positive_closure_projection_applied":projection_applied,
        "positive_current_trials":bool(positive_trials),
        "evaluation_limit_note":"Solver budget plus at most two final residual/projection checks"}


def reconstruct_frame(row, frame, response, solver, *, iterations=30, relaxation=0.2,
                      ion_pressure_ratio=1.0, profile_exponent=1.0, tf_turns=24,
                      solver_method="picard", initial_current=None, max_evaluations=400,
                      max_seconds=15.0, solver_tolerance=2e-3):
    """Return finite flux and transparent convergence/closure diagnostics.

    j_phi=[R*p_axis*(alpha+1)/delta_psi + A/R]*s**alpha, s=(psi-psi_edge)/delta.
    A enforces integral j_phi dR dZ=Ip exactly. p=p_axis*s**(alpha+1).
    This is an explicit assumed GS closure; unknown ion/fast pressure and FF' are
    not measured. Failed Picard convergence must never be labelled a solved equilibrium.
    The 24 effective TF turns follow the original MAST central-rod description:
    https://etheses.whiterose.ac.uk/id/eprint/14209/1/516649.pdf . This is not a
    MAST-U machine substitution; all poloidal geometry comes from the public row.
    """
    time = float(row["efit_times"][frame])
    vacuum, coverage = response.at(row, np.array([time]))
    vacuum = vacuum[0]
    ip, ip_mask = interpolate_signal(row["magnetics_time"], row["magnetics_plasma_current"], np.array([time]))
    ip = float(ip[0]) * 1000
    if ip <= 0 or not ip_mask[0] or not coverage.all():
        raise ValueError("This initial prototype requires observed coils and positive Ip")
    mask_path = SCORER / "masks" / "mast_envelope.npz"
    with np.load(mask_path) as data:
        envelope = data["mask_coarse"].astype(bool)
        if envelope.shape != solver.shape:
            raise ValueError("Published MAST envelope shape does not match the field grid")
    tsR, pe, valid, observed = thomson_at(row, time)
    if np.count_nonzero(pe > 0) < 8:
        raise ValueError("Insufficient observed positive Thomson channels for this closure")
    paxis = float(np.quantile(pe[pe > 0], 0.9)) * (1 + ion_pressure_ratio)
    r0 = float(np.sum(tsR * pe) / np.sum(pe))
    support = tsR[pe > 0.05 * np.max(pe)]
    a = float(np.clip((support.max() - support.min()) / 2, 0.25, 0.65))
    rho2 = ((solver.RR - r0) / a)**2 + (solver.ZZ / (1.8 * a))**2
    jtor = np.maximum(0, 1 - rho2) ** profile_exponent * envelope
    jtor.ravel()[solver.boundary] = 0
    jtor *= ip / (jtor.sum() * solver.dR * solver.dZ)
    continuation_used = initial_current is not None
    if continuation_used:
        initial_current = np.asarray(initial_current,dtype=float)
        if initial_current.shape != solver.shape or not np.isfinite(initial_current).all():
            raise ValueError("Continuation requires a finite current on the same physical grid")
        jtor = initial_current.copy()
        jtor[~envelope] = 0
        jtor.ravel()[solver.boundary] = 0
        if jtor.sum() <= 0:
            raise ValueError("Continuation current must have a positive integral")
        jtor *= ip/(jtor.sum()*solver.dR*solver.dZ)
    plasma = solver.solve(jtor)
    initial = vacuum + plasma
    jtor, nonlinear = solve_profile_current(vacuum,jtor,solver,envelope,ip,paxis,profile_exponent,
        method=solver_method,iterations=iterations,relaxation=relaxation,
        max_evaluations=max_evaluations,max_seconds=max_seconds,tolerance=solver_tolerance)
    plasma = solver.solve(jtor)
    psi = vacuum + plasma
    surface = _surface(psi, solver, envelope)
    diagnostics = {
        "time_ms": time, "input_Ip_A": ip, "integrated_Ip_A": float(jtor.sum()*solver.dR*solver.dZ),
        "linear_GS_relative_residual": solver.residual(plasma, jtor),
        **nonlinear, "continuation_used":continuation_used,
        "closed_surface": surface is not None, "paxis_assumption_Pa": paxis,
        "initial_TS_center_R_m": r0, "initial_minor_radius_m": a,
        "TS_observed_positive_fraction": float(observed.mean()),
        "TS_positive_available_fraction": float(valid.mean()),
        "TS_endpoint_extrapolation": bool(np.any(valid & ~observed)),
        "TS_endpoint_policy": "Nearest finite endpoint input value, explicitly flagged outside native coverage",
        "q95": None, "betaN": None,
        "flux_min": float(psi.min()), "flux_max": float(psi.max()),
        "gauge": "free-space R=0/infinity; unverified against EFIT target gauge",
        "Ip_relative_error":float(abs(jtor.sum()*solver.dR*solver.dZ-ip)/ip),
        "current_min_A_per_m2":float(jtor.min()),
        "current_negative_mass_fraction":float(-np.minimum(jtor,0).sum()/max(np.abs(jtor).sum(),1e-30)),
    }
    if surface is not None:
        C, axis, edge, delta, inside = surface
        envelope_on_C = _bilinear(envelope.astype(float), solver.R, solver.Z, C[:,0], C[:,1])
        diagnostics["LCFS_envelope_contact_fraction"] = float(np.mean(envelope_on_C < 0.75))
        from fusion.mast_structure import check_separatrix
        structure=check_separatrix(psi,solver.R,solver.Z,C,edge,delta,row)
        diagnostics["structure"]=structure
        diagnostics["diverted_topology_verified"] = bool(structure["separatrix_supported"] and
                                                          diagnostics["LCFS_envelope_contact_fraction"]<0.01)
        diagnostics["LCFS_limitation"] = "Native-grid saddle support is a structural check, not hidden-label accuracy"
        diagnostics["derived"] = derive_frame(psi, solver.R, solver.Z, "MAST", envelope,
                                              envelope.astype(float), contour=C, axis_sign=1.0)
        tf, _ = interpolate_signal(row["magnetics_time"], row["magnetics_tf_current"], np.array([time]))
        Fvac = abs(MU0 / (2*np.pi) * tf_turns * float(tf[0]) * 1000)
        s = np.clip((psi-edge)/delta, 0, 1)
        profile = s**profile_exponent * inside
        jp = solver.RR * paxis * (profile_exponent+1) / delta * profile
        A = (ip-jp.sum()*solver.dR*solver.dZ) / ((profile/solver.RR).sum()*solver.dR*solver.dZ)
        F95sq = Fvac**2 + 2*MU0*A*delta/(profile_exponent+1)*0.05**(profile_exponent+1)
        diagnostics["F_squared_min_T2_m2"] = float(min(Fvac**2,Fvac**2+2*MU0*A*delta/(profile_exponent+1)))
        q_level = edge + 0.05*delta
        for c in find_contours(psi, q_level):
            if is_closed(c) and encloses_point(c, axis[2], axis[3]):
                cq = grid_to_physical(c, solver.R, solver.Z)
                dz, dr = np.gradient(psi, solver.Z, solver.R)
                bp = np.sqrt(dz**2+dr**2)/solver.RR
                values = 1/(cq[:, 0]**2 * np.maximum(_bilinear(bp,solver.R,solver.Z,cq[:,0],cq[:,1]),1e-12))
                dl = np.linalg.norm(np.roll(cq,-1,axis=0)-cq,axis=1)
                if F95sq > 0:
                    diagnostics["q95"] = float(np.sqrt(F95sq)/(2*np.pi)*np.sum(0.5*(values+np.roll(values,-1))*dl))
                break
        pressure = paxis*s**(profile_exponent+1)
        pmean = float(np.sum(pressure[inside]*solver.RR[inside])/np.sum(solver.RR[inside]))
        Rgeom = 0.5*(C[:,0].max()+C[:,0].min())
        aminor = 0.5*(C[:,0].max()-C[:,0].min())
        Bt = Fvac/Rgeom
        if Bt > 0:
            diagnostics["betaN"] = float(100 * 2*MU0*pmean/Bt**2 * aminor*Bt/(ip/1e6))
        diagnostics["Fvac_T_m"] = Fvac
        diagnostics["scalar_status"] = "approximate closure outputs; not calibrated or validated on MAST labels"
        psi_chord=_bilinear(psi,solver.R,solver.Z,tsR,np.zeros_like(tsR))
        predicted_pe=paxis/(1+ion_pressure_ratio)*np.clip((psi_chord-edge)/delta,0,1)**(profile_exponent+1)
        diagnostics["TS_pressure_relative_RMSE"]=float(np.linalg.norm(predicted_pe[valid]-pe[valid])/
                                                      max(np.linalg.norm(pe[valid]),1e-30))
    diagnostics["scalar_valid"] = bool(diagnostics["q95"] is not None and diagnostics["betaN"] is not None
                                        and np.isfinite(diagnostics["q95"]) and np.isfinite(diagnostics["betaN"])
                                        and diagnostics.get("F_squared_min_T2_m2",-1)>0)
    diagnostics["structurally_admissible_equilibrium"] = bool(diagnostics["converged"]
        and diagnostics.get("diverted_topology_verified",False) and diagnostics["scalar_valid"]
        and diagnostics["current_negative_mass_fraction"]<1e-4)
    return {"psi": psi.astype(np.float32), "vacuum": vacuum.astype(np.float32),
            "initial_psi": initial.astype(np.float32), "jtor": jtor.astype(np.float32),
            "diagnostics": diagnostics}
