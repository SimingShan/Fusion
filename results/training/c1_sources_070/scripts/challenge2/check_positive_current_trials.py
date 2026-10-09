#!/usr/bin/env python3
"""Check trial-domain projection, actual residuals and unchanged default inference."""
import argparse
import ast
import copy
import hashlib
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from fusion.mast_physics import project_current_simplex, solve_profile_current
from scripts.codabench_submit import write_json


class RemoveDisabledProjection(ast.NodeTransformer):
    def visit_FunctionDef(self, node):
        node = self.generic_visit(node)
        if node.args.args and node.args.args[-1].arg == "positive_current_trials":
            assert isinstance(node.args.defaults[-1], ast.Constant) and node.args.defaults[-1].value is False
            node.args.args.pop(); node.args.defaults.pop()
        for index in reversed(range(len(node.args.kwonlyargs))):
            if node.args.kwonlyargs[index].arg == "positive_trials":
                default = node.args.kw_defaults[index]
                assert isinstance(default, ast.Constant) and default.value is False
                node.args.kwonlyargs.pop(index); node.args.kw_defaults.pop(index)
        return node

    def visit_If(self, node):
        if isinstance(node.test, ast.Name) and node.test.id == "positive_trials":
            assert not node.orelse
            return None
        return self.generic_visit(node)

    def visit_Call(self, node):
        node = self.generic_visit(node)
        if isinstance(node.func, ast.Name) and node.func.id == "infer" and node.args:
            if isinstance(node.args[-1], ast.Name) and node.args[-1].id == "positive_current_trials":
                node.args.pop()
        retained = []
        for kw in node.keywords:
            if kw.arg == "positive_trials":
                assert isinstance(kw.value, ast.Name) and kw.value.id == "positive_current_trials"
            else:
                retained.append(kw)
        node.keywords = retained
        return node

    def visit_Dict(self, node):
        node = self.generic_visit(node)
        pairs = [(k, v) for k,v in zip(node.keys, node.values)
                 if not (isinstance(k, ast.Constant) and k.value == "positive_current_trials")]
        node.keys, node.values = [k for k,v in pairs], [v for k,v in pairs]
        return node


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise ValueError("Preserve previous checks")
    for path, names in [("fusion/mast_physics.py", ("solve_profile_current",)),
                        ("scripts/challenge2/evaluate_profile_prior.py", ("infer", "infer_with_retry", "valid_prediction"))]:
        old = ast.parse((ROOT / "results/challenge2/c2_export_045/source_snapshot" / path).read_text())
        new = ast.parse((ROOT / path).read_text())
        for name in names:
            before = next(x for x in old.body if isinstance(x, ast.FunctionDef) and x.name == name)
            after = next(x for x in new.body if isinstance(x, ast.FunctionDef) and x.name == name)
            assert ast.dump(before) == ast.dump(RemoveDisabledProjection().visit(copy.deepcopy(after))), name
    shape = (5, 5)
    active = np.zeros(shape, bool); active[1:-1, 1:-1] = True
    boundary = ~active.ravel()
    solver = SimpleNamespace(shape=shape, boundary=boundary, dR=.2, dZ=.3)
    ip, area = 13., solver.dR*solver.dZ
    physical_scale = ip/area
    current = np.full(shape, -2.)
    uniform = project_current_simplex(current, active, solver, ip)
    np.testing.assert_allclose(uniform[active], physical_scale/active.sum(), rtol=1e-14)
    assert np.count_nonzero(uniform[~active]) == 0
    rng = np.random.default_rng(2083)
    worst = 0.
    for _ in range(100):
        values = rng.normal(size=active.sum())*2
        current = rng.normal(size=shape)*physical_scale
        current[active] = values*physical_scale
        projected = project_current_simplex(current, active, solver, ip)
        x = projected[active]/physical_scale
        reference = minimize(lambda v: .5*np.sum((v-values)**2), np.full(len(values), 1/len(values)),
            jac=lambda v: v-values, bounds=[(0., None)]*len(values), method="SLSQP",
            constraints={"type": "eq", "fun": lambda v: v.sum()-1, "jac": lambda v: np.ones_like(v)},
            options={"ftol": 1e-12, "maxiter": 500})
        assert reference.success, reference.message
        error = abs(float(np.sum((x-values)**2)-np.sum((reference.x-values)**2)))
        worst = max(worst, error)
        assert error < 1e-8 and x.min() >= 0
        np.testing.assert_allclose(projected.sum()*area, ip, rtol=1e-14)
        np.testing.assert_allclose(project_current_simplex(projected, active, solver, ip), projected, rtol=1e-13, atol=1e-11)
        shifted = current.copy(); shifted[active] += 123*physical_scale
        np.testing.assert_allclose(project_current_simplex(shifted, active, solver, ip), projected, rtol=1e-11, atol=1e-9)
    target = np.zeros(shape)
    target[active] = np.arange(1, active.sum()+1)
    target *= ip/(target.sum()*area)
    observed = []

    def solve(current):
        assert current.min() >= 0 and np.all(current[~active] == 0)
        np.testing.assert_allclose(current.sum()*area, ip, rtol=1e-13)
        observed.append(current.copy())
        return current.copy()

    solver.solve = solve
    closure = lambda field: .3*field+.7*target
    initial = np.zeros(shape); initial[active] = np.linspace(-8, 4, active.sum())*physical_scale
    solved, info = solve_profile_current(np.zeros(shape), initial, solver, active, ip, 1., 1.,
        method="anderson", iterations=100, max_evaluations=400, max_seconds=30, tolerance=1e-8,
        profile_function=closure, positive_trials=True)
    assert info["converged"] and observed
    independent = float(np.linalg.norm(closure(solved)-solved)/np.linalg.norm(solved))
    assert independent < 1e-8
    np.testing.assert_allclose(independent, info["nonlinear_profile_relative_residual"], rtol=1e-5, atol=1e-12)
    np.testing.assert_allclose(solved, target, rtol=1e-7, atol=1e-7)
    stopped, capped = solve_profile_current(np.zeros(shape), initial, solver, active, ip, 1., 1.,
        method="anderson", iterations=100, max_evaluations=1, max_seconds=30, tolerance=1e-8,
        profile_function=closure, positive_trials=True)
    actual = float(np.linalg.norm(closure(stopped)-stopped)/np.linalg.norm(stopped))
    assert not capped["converged"] and actual > 1e-8
    np.testing.assert_allclose(actual, capped["nonlinear_profile_relative_residual"], rtol=1e-13)
    for value in (0., -1., np.nan):
        try:
            project_current_simplex(initial, active, solver, value)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid measured current accepted")
    result = {"passed": True, "disabled_option_ast_identical": True, "independent_projection_cases": 100,
        "maximum_objective_difference": worst, "negative_input_case_passed": True, "nonnegative_trial_fields": True,
        "exact_measured_integral": True, "projection_idempotence_and_shift_invariance": True,
        "independently_recomputed_physical_residual": independent, "stopped_run_not_mislabelled_converged": True,
        "source_sha256": {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
            ("fusion/mast_physics.py", "scripts/challenge2/evaluate_profile_prior.py", str(Path(__file__).relative_to(ROOT)))},
        "synthetic_only": True, "target_data_read": False, "eligible_for_submission": False}
    write_json(a.output, result)
    print({k:v for k,v in result.items() if k != "source_sha256"})


if __name__ == "__main__":
    main()
