#!/usr/bin/env python3
"""Independent optimization and physical-closure checks; no target data read."""
import argparse
import ast
import copy
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
from scipy.optimize import minimize, nnls

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from fusion.current_constrained_pressure import bounded_nnls, CurrentConstrainedPressure, pressure_current_weights
from fusion.measured_pressure import PressureConstraint
from fusion.smooth_profiles import current, integral
from scripts.codabench_submit import write_json


class RemoveDisabledOption(ast.NodeTransformer):
    def visit_FunctionDef(self, node):
        node = self.generic_visit(node)
        if node.args.args[-1].arg == "constrain_pressure_current":
            node.args.args.pop(); node.args.defaults.pop()
        return node

    def visit_If(self, node):
        if isinstance(node.test, ast.Name) and node.test.id == "constrain_pressure_current":
            return None
        return self.generic_visit(node)

    def visit_Call(self, node):
        node = self.generic_visit(node)
        if (isinstance(node.func, ast.Name) and node.func.id == "infer" and node.args
                and isinstance(node.args[-1], ast.Name) and node.args[-1].id == "constrain_pressure_current"):
            node.args.pop()
        return node


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise ValueError("Use a fresh check output")
    previous = ast.parse((ROOT / "results/challenge2/c2_sources_033/scripts/challenge2/evaluate_profile_prior.py").read_text())
    changed = ast.parse((ROOT / "scripts/challenge2/evaluate_profile_prior.py").read_text())
    for name in ("infer", "infer_with_retry", "valid_prediction"):
        old = next(x for x in previous.body if isinstance(x, ast.FunctionDef) and x.name == name)
        new = next(x for x in changed.body if isinstance(x, ast.FunctionDef) and x.name == name)
        normalized = RemoveDisabledOption().visit(copy.deepcopy(new))
        assert ast.dump(old) == ast.dump(normalized), name
    np.testing.assert_allclose(bounded_nnls(np.eye(4), np.ones(4), np.ones(4), 2), .5, atol=1e-14)
    rng = np.random.default_rng(2082)
    worst = 0.
    for _ in range(100):
        A = np.vstack((rng.normal(size=(15, 4)), .2*np.eye(4)))
        b, w = rng.normal(size=19), rng.uniform(.1, 1., size=4)
        ordinary = nnls(A, b)[0]
        limit = max(float(w @ ordinary)*.5, .01)
        x = bounded_nnls(A, b, w, limit)
        reference = minimize(lambda v: .5*np.sum((A @ v-b)**2), np.zeros(4),
            jac=lambda v: A.T @ (A @ v-b), bounds=[(0, None)]*4,
            constraints={"type": "ineq", "fun": lambda v: limit-w @ v, "jac": lambda v: -w},
            method="SLSQP", options={"ftol": 1e-12, "maxiter": 500})
        assert reference.success, reference.message
        assert x.min() >= 0 and w @ x <= limit+1e-10
        error = abs(float(np.sum((A @ x-b)**2)-np.sum((A @ reference.x-b)**2)))
        worst = max(error, worst)
        assert error < 1e-8
        if w @ ordinary <= limit:
            np.testing.assert_array_equal(x, ordinary)
    R, Z = np.linspace(.7, 2.7, 129), np.linspace(-1, 1, 129)
    RR, ZZ = np.meshgrid(R, Z)
    r0, radius, delta = 1.7, .4, .2
    rho = ((RR-r0)**2+ZZ**2)/radius**2
    phi, inside = delta*(1-rho), rho <= 1
    angle = np.linspace(0, 2*np.pi, 257)
    contour = np.column_stack((r0+radius*np.cos(angle), radius*np.sin(angle)))
    surface = contour, None, 0., delta, inside
    solver = SimpleNamespace(R=R, Z=Z, RR=RR, ZZ=ZZ, dR=R[1]-R[0], dZ=Z[1]-Z[0])
    coords = np.column_stack((R[44:85], np.zeros(41)))
    s = 1-((coords[:, 0]-r0)/radius)**2
    observations = {"positions": coords, "pressure": 5000*s**2, "observed": np.ones(41, bool)}
    prior = np.array([.1, .2, .3, .4])
    base = PressureConstraint(observations, 3e4, prior)
    paxis, shape, report = base.fit(phi, solver, None, surface)
    weights = pressure_current_weights(phi, solver, surface, 2.)
    coeff = paxis/2*shape/(integral(1.) @ shape)
    pressure_ip = float(weights @ coeff)
    # Independently integrate the pressure term in the existing closure.
    from fusion.smooth_profiles import basis
    pprime = paxis/delta*(basis(np.clip(phi/delta, 0, 1)) @ shape)/(integral(1.) @ shape)
    explicit = (RR-contour[:, 0].min()**2/RR)*pprime*inside
    np.testing.assert_allclose(explicit.sum()*solver.dR*solver.dZ, pressure_ip, rtol=1e-14)
    inactive = CurrentConstrainedPressure(observations, 3e4, prior, ip=pressure_ip*2)
    pa, sh, diagnostic = inactive.fit(phi, solver, None, surface)
    assert pa == paxis and not diagnostic["pressure_current_bound_active"]
    np.testing.assert_array_equal(sh, shape)
    ip = pressure_ip*.4
    active = CurrentConstrainedPressure(observations, 3e4, prior, ip=ip)
    pa, sh, diagnostic = active.fit(phi, solver, None, surface)
    assert diagnostic["pressure_current_bound_active"]
    density = current(phi, solver, None, ip, pa, sh, np.ones(4), surface=surface)
    assert density.min() >= 0
    np.testing.assert_allclose(density.sum()*solver.dR*solver.dZ, ip, rtol=1e-13)
    shifted = active.fit(phi+2.5, solver, None, (contour, None, 2.5, delta, inside))
    np.testing.assert_allclose(shifted[0], pa, rtol=1e-10)
    np.testing.assert_allclose(shifted[1], sh, rtol=1e-9, atol=1e-8)
    missing = {"positions": np.empty((0, 2)), "pressure": np.empty(0), "observed": np.empty(0, bool)}
    pa, sh, diagnostic = CurrentConstrainedPressure(missing, 3e4, prior, ip=ip).fit(phi, solver, None, surface)
    assert not diagnostic["pressure_fit_used"] and diagnostic["pressure_prior_amplitude_constrained"]
    density = current(phi, solver, None, ip, pa, sh, np.ones(4), surface=surface)
    assert density.min() >= 0
    np.testing.assert_allclose(density.sum()*solver.dR*solver.dZ, ip, rtol=1e-13)
    rejected = 0
    for w, limit in [(np.full(4, -1.), 1.), (np.ones(4), 0.), (np.full(4, np.nan), 1.)]:
        try:
            bounded_nnls(np.eye(4), np.ones(4), w, limit)
        except ValueError:
            rejected += 1
        else:
            raise AssertionError("Invalid current bound accepted")
    result = {"passed": True, "disabled_option_ast_identical": True, "independent_optimization_cases": 100,
              "maximum_objective_difference": worst, "analytic_solution_passed": True,
              "current_integral_identity_passed": True, "inactive_constraint_bit_identical": True,
              "positive_current_and_exact_integral_passed": True, "gauge_invariance_passed": True,
              "insufficient_observations_prior_case_passed": True, "invalid_systems_rejected": rejected,
              "target_data_read": False, "synthetic_only": True}
    write_json(a.output, result)
    print(result)


if __name__ == "__main__":
    main()
