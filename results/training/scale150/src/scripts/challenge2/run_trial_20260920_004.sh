#!/bin/bash
set -euo pipefail
cd "${FUSION_ROOT:?set FUSION_ROOT to the package root}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
${PYTHON:-python} scripts/challenge2/check_measured_pressure.py
${PYTHON:-python} -u scripts/challenge2/evaluate_profile_prior.py \
  --prior results/challenge2/c2_prior_002/prior.joblib \
  --output results/challenge2/c2_evaluation_004 \
  --shots 8 --frames 3 --max-seconds 10 \
  --measured-pressure --pressure-factor 2
