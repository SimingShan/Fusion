"""Fit measured pressure inside the current budget of the existing closure.

This is an opt-in constraint on pressure coefficients, not on the predicted
field. The available current is a supplied magnetic measurement. The returned
coefficients still enter the same field solve and scalar integration.
"""
import itertools

import numpy as np
from scipy.optimize import nnls

from fusion.measured_pressure import PressureConstraint, _bilinear, _surface
from fusion.smooth_profiles import basis, integral


def bounded_nnls(A, b, weights, limit):
    """Solve min ||A x-b||², x>=0 and weights@x<=limit, for four columns.

If the ordinary nonnegative fit violates the extra bound, the optimum lies on
that bound. Enumerating its nonzero supports solves each equality-constrained
quadratic exactly; no nonlinear optimization stopping flag is used.
    """
    A, b, weights = np.asarray(A, float), np.asarray(b, float), np.asarray(weights, float)
    if (A.ndim != 2 or A.shape[1] != 4 or b.shape != (A.shape[0],)
            or weights.shape != (4,) or not np.isfinite(A).all()
            or not np.isfinite(b).all() or not np.isfinite(weights).all()
            or np.any(weights < 0) or not np.isfinite(limit) or limit <= 0):
        raise ValueError("Invalid constrained pressure system")
    ordinary = nnls(A, b, maxiter=500)[0]
    if weights @ ordinary <= limit:
        return ordinary
    best, objective = None, np.inf
    for count in range(1, 5):
        for subset in itertools.combinations(range(4), count):
            ids = np.asarray(subset)
            a, w = A[:, ids], weights[ids]
            if not np.any(w):
                continue
            H, g = a.T @ a, a.T @ b
            try:
                unconstrained = np.linalg.solve(H, g)
                direction = np.linalg.solve(H, w)
            except np.linalg.LinAlgError:
                continue
            denominator = w @ direction
            if denominator <= 0:
                continue
            multiplier = (w @ unconstrained-limit)/denominator
            values = unconstrained-multiplier*direction
            if multiplier < -1e-10 or np.any(values < -1e-10):
                continue
            candidate = np.zeros(4)
            candidate[ids] = np.maximum(values, 0.)
            if weights @ candidate > limit*(1+1e-9):
                continue
            error = float(np.sum((A @ candidate-b)**2))
            if error < objective:
                best, objective = candidate, error
    if best is None:
        raise ValueError("No feasible constrained pressure fit")
    return best


def pressure_current_weights(phi, solver, surface, factor):
    """Linear current integral per unit electron-pressure coefficient [A/Pa]."""
    contour, _, edge, delta, inside = surface
    s = np.clip((phi-edge)/delta, 0, 1)
    inner = float(contour[:, 0].min())
    radial = (solver.RR-inner**2/solver.RR)*inside
    weights = factor/delta*np.sum(basis(s)*radial[..., None], axis=(0, 1))*solver.dR*solver.dZ
    if not np.isfinite(weights).all() or np.any(weights < -1e-10):
        raise ValueError("Invalid pressure-current coefficients")
    return np.maximum(weights, 0.)


class CurrentConstrainedPressure(PressureConstraint):
    def __init__(self, *args, ip, **kwargs):
        super().__init__(*args, **kwargs)
        if not np.isfinite(ip) or ip <= 0 or self.regularization <= 0:
            raise ValueError("Positive measured current and fit regularization required")
        self.ip = float(ip)

    def fit(self, phi, solver, envelope, surface=None):
        surface = _surface(phi, solver, envelope) if surface is None else surface
        paxis, shape, report = super().fit(phi, solver, envelope, surface)
        weights = pressure_current_weights(phi, solver, surface, self.factor)
        coef = paxis/self.factor*shape/(integral(1.) @ shape)
        current = float(weights @ coef)
        report.update(pressure_current_bound_active=False, unconstrained_pressure_current_A=current,
                      measured_current_constraint_A=self.ip)
        # A tiny interior margin avoids a negative closure amplitude caused only
        # by roundoff after reassembling the same linear pressure integral.
        limit = self.ip*(1-1e-10)
        if current <= limit:
            return paxis, shape, report
        if not report["pressure_fit_used"]:
            # With insufficient observations the prior shape is fixed. Its only
            # free parameter here is amplitude, whose admissible maximum is exact.
            scale = limit/current
            report.update(pressure_current_bound_active=True, constrained_pressure_current_A=limit,
                          pressure_prior_amplitude_constrained=True)
            return paxis*scale, shape, report
        coords, pressure = self.observations["positions"], self.observations["pressure"]
        valid = ((coords[:, 0] >= solver.R[0]) & (coords[:, 0] <= solver.R[-1]) &
                 (coords[:, 1] >= solver.Z[0]) & (coords[:, 1] <= solver.Z[-1]))
        coords, pressure = coords[valid], pressure[valid]
        values = _bilinear(phi, solver.R, solver.Z, coords[:, 0], coords[:, 1])
        _, _, edge, delta, _ = surface
        s = np.clip((values-edge)/delta, 0, 1)
        reference = max(float(np.quantile(pressure, .9)), 1.)
        B, y = integral(s), pressure/reference
        prior = self.prior_axis/self.factor*self.prior_shape/(integral(1.) @ self.prior_shape)/reference
        scale = np.maximum(np.linalg.norm(B, axis=0), 1e-5)
        root = np.sqrt(self.regularization)
        A, rhs = np.vstack((B/scale, root*np.eye(4))), np.r_[y, root*prior*scale]
        normalized_weights = weights*reference/scale/self.ip
        coef = bounded_nnls(A, rhs, normalized_weights, limit/self.ip)/scale*reference
        paxis = self.factor*float(integral(1.) @ coef)
        if paxis <= 0 or not np.isfinite(coef).all():
            raise ValueError("Invalid constrained pressure coefficients")
        fitted_current = float(weights @ coef)
        if fitted_current > self.ip:
            raise ValueError("Constrained pressure exceeded the measured current")
        report.update(pressure_current_bound_active=True, constrained_pressure_current_A=fitted_current,
                      pressure_prior_amplitude_constrained=False,
                      pressure_relative_rmse=float(np.linalg.norm(B @ coef-pressure)/max(np.linalg.norm(pressure), 1e-20)))
        return paxis, coef, report
