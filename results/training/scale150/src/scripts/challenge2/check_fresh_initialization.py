#!/usr/bin/env python3
"""Verify preservation of the frozen predictor and validity-only recovery."""
import argparse
import ast
import copy
import hashlib
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.challenge2 import evaluate_profile_prior as evaluator
from scripts.challenge2.check_positive_current_trials import RemoveDisabledProjection
from scripts.codabench_submit import write_json


def dump(node):
    return ast.dump(node, include_attributes=False)


def function(tree, name):
    found = [x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == name]
    if len(found) != 1:
        raise ValueError(f"Expected one function: {name}")
    return copy.deepcopy(found[0])


class ReceiptOnly(ast.NodeTransformer):
    fields = {"initialization_retry_frames", "initialization_retry_used",
              "original_continuation_failure", "initialization_retry_failure"}

    def visit_Tuple(self, node):
        node.elts = [x for x in node.elts if not (isinstance(x, ast.Constant) and x.value in self.fields)]
        return self.generic_visit(node)

    def visit_Dict(self, node):
        pairs = [(k, v) for k, v in zip(node.keys, node.values)
                 if not (isinstance(k, ast.Constant) and k.value in self.fields)]
        node.keys, node.values = [k for k, v in pairs], [v for k, v in pairs]
        return self.generic_visit(node)


def prove_unchanged(parent, source):
    relative = "scripts/challenge2/evaluate_profile_prior.py"
    old, new = [ast.parse((p / relative).read_text()) for p in (parent, source)]
    prefix = lambda tree: ast.Module(body=tree.body[:next(i for i, x in enumerate(tree.body)
        if isinstance(x, ast.FunctionDef))], type_ignores=[])
    if dump(prefix(old)) != dump(prefix(new)):
        raise ValueError("Prediction imports or module initialization changed")
    for name in ("infer", "infer_with_retry", "valid_prediction"):
        if dump(function(old, name)) != dump(RemoveDisabledProjection().visit(function(new, name))):
            raise ValueError(f"Existing prediction function changed: {name}")
    old_physics, new_physics = [ast.parse((p / "fusion/mast_physics.py").read_text()) for p in (parent, source)]
    new_physics.body = [x for x in new_physics.body if not (
        isinstance(x, ast.FunctionDef) and x.name == "project_current_simplex")]
    if dump(old_physics) != dump(RemoveDisabledProjection().visit(new_physics)):
        raise ValueError("Existing numerical implementation changed")
    wrapper = function(new, "infer_with_restart")
    if dump(wrapper.args) != dump(function(new, "infer_with_retry").args):
        raise ValueError("Recovery defaults differ from the existing policy")
    expected = ast.parse('''
options = dict(max_seconds=max_seconds, max_evaluations=max_evaluations,
    measured_pressure=measured_pressure, pressure_factor=pressure_factor,
    observed_pressure_only=observed_pressure_only, solver_method=solver_method,
    constrain_pressure_current=constrain_pressure_current, positive_current_trials=positive_current_trials)
original, failure = None, None
''').body
    if [dump(x) for x in wrapper.body[1:3]] != [dump(x) for x in expected]:
        raise ValueError("Recovery changes inputs or settings before its original attempt")
    expected = ast.parse('''
original = infer_with_retry(row, frame, prior, solver, envelope, previous, **options)
if valid_prediction(original) or previous is None:
    return original
''').body
    if (not isinstance(wrapper.body[3], ast.Try)
            or [dump(x) for x in wrapper.body[3].body[:2]] != [dump(x) for x in expected]):
        raise ValueError("Valid original results are not returned unchanged")
    exporters = [ast.parse((p / "scripts/challenge2/export_mast_predictions.py").read_text())
                 for p in (parent, source)]
    before, after = [function(t, "predict_shot") for t in exporters]
    if dump(before) != dump(ReceiptOnly().visit(after)):
        raise ValueError("Native frame order, prediction inputs or previous-state updates changed")
    before, after = [function(t, "initialize") for t in exporters]
    expected_branch = ast.parse('''
if plan.get("retry_fresh_initialization"):
    if not plan.get("retry_observed_pressure"):
        raise ValueError("Fresh initialization requires the existing conditional pressure policy")
    infer=infer_with_restart
''').body[0]
    found = 0
    for node in list(after.body):
        if isinstance(node, ast.ImportFrom) and node.module == "scripts.challenge2.evaluate_profile_prior":
            if [x.name for x in node.names] != ["infer", "infer_with_retry", "infer_with_restart"]:
                raise ValueError("Unexpected predictor imports")
            node.names.pop()
        if dump(node) == dump(expected_branch):
            after.body.remove(node)
            found += 1
    if found != 1 or dump(before) != dump(after):
        raise ValueError("Unexpected export dispatch or initialization")
    return {"original_prediction_ast_equal": True, "original_numerics_ast_equal": True,
            "valid_original_returned_unchanged": True, "native_sequence_ast_equal": True,
            "only_invalid_continuations_can_change": True}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise ValueError("Preserve previous check evidence")
    checks = prove_unchanged(ROOT / "results/challenge2/c2_export_045/source_snapshot", ROOT)
    previous = np.ones((2, 2))
    valid = lambda: (np.ones((2, 2)), np.ones((2, 2)), {"scalar_valid": True, "q95": 2., "betaN": 1.})
    invalid = lambda: (np.ones((2, 2)), np.ones((2, 2)), {"scalar_valid": False, "status": "invalid"})
    row, prior, solver, envelope = object(), object(), object(), object()
    options = dict(max_seconds=317, max_evaluations=619, measured_pressure=True, pressure_factor=2.3,
                   observed_pressure_only=False, solver_method="anderson",
                   constrain_pressure_current=False, positive_current_trials=False)
    cases = [([valid()], previous, True, 1), ([invalid()], None, False, 1),
             ([invalid(), valid()], previous, True, 2),
             ([ValueError("original"), valid()], previous, True, 2),
             ([invalid(), invalid()], previous, False, 2),
             ([invalid(), FloatingPointError("fresh")], previous, False, 2),
             ([ValueError("original"), ValueError("fresh")], previous, None, 2),
             ([ValueError("original")], None, None, 1)]
    for sequence, state, expected_valid, calls in cases:
        with patch.object(evaluator, "infer_with_retry", side_effect=sequence) as mock:
            try:
                result = evaluator.infer_with_restart(row, 7, prior, solver, envelope, state, **options)
            except ValueError:
                assert expected_valid is None
            else:
                assert evaluator.valid_prediction(result) is expected_valid
                if calls == 1 or not expected_valid:
                    assert result is sequence[0]
                else:
                    assert result is sequence[1] and result[2]["initialization_retry_used"]
            assert mock.call_count == calls
            for i, call in enumerate(mock.call_args_list):
                assert call.args[:5] == (row, 7, prior, solver, envelope)
                assert call.args[5] is (state if i == 0 else None)
                assert call.kwargs == options
    rejected = [valid(), valid(), valid()]
    rejected[0][2]["q95"] = -1.
    rejected[1][2]["betaN"] = float("nan")
    rejected[2][0][0, 0] = float("inf")
    assert not any(evaluator.valid_prediction(x) for x in rejected)
    files = ("scripts/challenge2/evaluate_profile_prior.py", "scripts/challenge2/export_mast_predictions.py",
             "fusion/mast_physics.py", "scripts/challenge2/check_positive_current_trials.py",
             str(Path(__file__).relative_to(ROOT)))
    result = {"passed": True, **checks, "recovery_cases": len(cases), "invalid_scalar_and_field_checks": 3,
              "source_sha256": {f: hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files},
              "synthetic_only": True, "target_data_read": False, "eligible_for_submission": False}
    write_json(a.output, result)
    print({k: v for k, v in result.items() if k != "source_sha256"})


if __name__ == "__main__":
    main()
