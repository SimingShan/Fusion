#!/bin/bash
set -euo pipefail
cd "${FUSION_ROOT:?set FUSION_ROOT to the package root}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
${PYTHON:-python} scripts/challenge2/merge_profile_shards.py \
  --source results/challenge2/c2_native_development_036 --output results/challenge2/c2_evaluation_036 --parts 4
${PYTHON:-python} scripts/challenge2/summarize_native_coverage.py \
  --evaluation results/challenge2/c2_evaluation_036 --output results/challenge2/c2_evaluation_036/coverage_review.json
${PYTHON:-python} scripts/challenge2/score_profile_trials.py \
  --trials 29 36 --output results/challenge2/development_comparison_010
