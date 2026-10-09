#!/bin/bash
set -euo pipefail
cd "${FUSION_ROOT:?set FUSION_ROOT to the package root}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
${PYTHON:-python} -u scripts/challenge2/refit_affine_calibration.py \
  --data results/challenge2/c2_targets_002 --prior results/challenge2/c2_prior_002/prior.joblib \
  --output results/challenge2/c2_prior_006
${PYTHON:-python} -u scripts/challenge2/evaluate_profile_prior.py \
  --prior results/challenge2/c2_prior_006/prior.joblib --output results/challenge2/c2_evaluation_006 \
  --shots 8 --frames 3 --machines DIII-D --max-seconds 10 --measured-pressure --pressure-factor 2
${PYTHON:-python} -u scripts/challenge2/score_profile_trials.py \
  --trials 4 6 --output results/challenge2/development_comparison_004
