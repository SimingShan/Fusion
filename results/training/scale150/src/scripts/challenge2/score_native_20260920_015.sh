#!/bin/bash
set -euo pipefail
cd "${FUSION_ROOT:?set FUSION_ROOT to the package root}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
${PYTHON:-python} -u scripts/challenge2/merge_profile_shards.py \
  --source results/challenge2/c2_native_development_015 --parts 4 \
  --output results/challenge2/c2_evaluation_015
${PYTHON:-python} -u scripts/challenge2/score_profile_trials.py \
  --trials 15 --output results/challenge2/development_comparison_006
