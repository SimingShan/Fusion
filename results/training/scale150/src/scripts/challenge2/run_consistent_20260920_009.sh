#!/bin/bash
set -euo pipefail
cd "${FUSION_ROOT:?set FUSION_ROOT to the package root}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
${PYTHON:-python} -u scripts/challenge2/prepare_pressure_targets.py \
  --source results/challenge2/c2_targets_002 --output results/challenge2/c2_targets_009
${PYTHON:-python} -u scripts/challenge2/train_profile_prior.py \
  --data results/challenge2/c2_targets_009 --output results/challenge2/c2_prior_009
${PYTHON:-python} -u scripts/challenge2/evaluate_profile_prior.py \
  --prior results/challenge2/c2_prior_009/prior.joblib --output results/challenge2/c2_evaluation_009 \
  --shots 8 --frames 3 --max-seconds 10 --measured-pressure --pressure-factor 2
${PYTHON:-python} -u scripts/challenge2/score_profile_trials.py \
  --trials 4 9 --output results/challenge2/development_comparison_005
