#!/bin/bash
set -euo pipefail
cd "${FUSION_ROOT:?set FUSION_ROOT to the package root}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
task_source=results/challenge2/c2_sources_033
${PYTHON:-python} -u "$task_source/scripts/challenge2/evaluate_profile_prior.py" \
  --prior results/challenge2/c2_prior_002/prior.joblib --output results/challenge2/c2_evaluation_033 \
  --data-root ${FUSION_ROOT}/data --shots 8 --frames 3 \
  --max-seconds 300 --max-evaluations 600 --measured-pressure --pressure-factor 2 \
  --prediction-jobs 1 --retry-observed-pressure --solver-method krylov
${PYTHON:-python} -u "$task_source/scripts/challenge2/evaluate_profile_prior.py" \
  --prior results/challenge2/c2_prior_002/prior.joblib --output results/challenge2/c2_evaluation_034 \
  --data-root ${FUSION_ROOT}/data --shots 8 --frames 3 \
  --max-seconds 300 --max-evaluations 600 --measured-pressure --pressure-factor 2 \
  --prediction-jobs 1 --retry-observed-pressure --solver-method anderson
${PYTHON:-python} scripts/challenge2/score_profile_trials.py \
  --trials 33 34 --output results/challenge2/development_comparison_009
