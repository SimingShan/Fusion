#!/bin/bash
set -euo pipefail
cd "${FUSION_ROOT:?set FUSION_ROOT to the package root}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
task_source=results/challenge2/c2_sources_022
${PYTHON:-python} -u "$task_source/scripts/challenge2/fit_pressure_factor.py" \
  --targets results/challenge2/c2_targets_002 --prior results/challenge2/c2_prior_002/prior.joblib \
  --data-root ${FUSION_ROOT}/data --output results/challenge2/c2_prior_023
${PYTHON:-python} -u "$task_source/scripts/challenge2/evaluate_profile_prior.py" \
  --prior results/challenge2/c2_prior_002/prior.joblib --output results/challenge2/c2_evaluation_022 \
  --data-root ${FUSION_ROOT}/data --shots 8 --frames 3 \
  --max-seconds 300 --max-evaluations 600 --measured-pressure --pressure-factor 2 --prediction-jobs 1
${PYTHON:-python} -u "$task_source/scripts/challenge2/evaluate_profile_prior.py" \
  --prior results/challenge2/c2_prior_023/prior.joblib --output results/challenge2/c2_evaluation_023 \
  --data-root ${FUSION_ROOT}/data --shots 8 --frames 3 \
  --max-seconds 300 --max-evaluations 600 --measured-pressure --pressure-factor 2 --prediction-jobs 1
${PYTHON:-python} scripts/challenge2/score_profile_trials.py \
  --trials 22 23 --output results/challenge2/development_comparison_007
