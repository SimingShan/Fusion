#!/bin/bash
set -euo pipefail
cd "${FUSION_ROOT:?set FUSION_ROOT to the package root}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
task_source=results/challenge2/c2_sources_042
${PYTHON:-python} -u "$task_source/scripts/challenge2/evaluate_profile_prior.py" \
  --prior results/challenge2/c2_prior_002/prior.joblib --output results/challenge2/c2_evaluation_042 \
  --data-root ${FUSION_ROOT}/data --shots 8 --frames 3 \
  --max-seconds 300 --max-evaluations 600 --measured-pressure --pressure-factor 2 \
  --prediction-jobs 1 --retry-observed-pressure --solver-method anderson
${PYTHON:-python} - <<'PY'
from pathlib import Path
import json
import numpy as np
from scripts.codabench_submit import write_json
old = Path('results/challenge2/c2_evaluation_034')
new = Path('results/challenge2/c2_evaluation_042')
a, b = [json.loads((x/'summary.json').read_text())['records'] for x in (old, new)]
assert len(a) == len(b) == 48
for before, after in zip(a, b):
    for key in ('machine', 'shot', 'frame', 'q95', 'betaN', 'scalar_valid', 'converged', 'nonlinear_profile_relative_residual'):
        assert before.get(key) == after.get(key), key
    name = f"{before['machine']}_{before['shot']}_{before['frame']:04d}.npz"
    with np.load(old/name) as x, np.load(new/name) as y:
        assert x.files == y.files
        for key in x.files:
            np.testing.assert_array_equal(x[key], y[key])
write_json(new/'default_parity.json', {'passed': True, 'frames': 48, 'prediction_arrays_bit_identical': True,
    'scalar_and_residual_values_identical': True, 'mast_labels_read': False, 'audit_labels_read': False})
print('Default prediction parity passed for all 48 frames', flush=True)
PY
${PYTHON:-python} -u "$task_source/scripts/challenge2/evaluate_profile_prior.py" \
  --prior results/challenge2/c2_prior_002/prior.joblib --output results/challenge2/c2_evaluation_043 \
  --data-root ${FUSION_ROOT}/data --shots 8 --frames 3 \
  --max-seconds 300 --max-evaluations 600 --measured-pressure --pressure-factor 2 \
  --prediction-jobs 1 --retry-observed-pressure --solver-method anderson --constrain-pressure-current
${PYTHON:-python} scripts/challenge2/score_profile_trials.py \
  --trials 42 43 --output results/challenge2/development_comparison_011
