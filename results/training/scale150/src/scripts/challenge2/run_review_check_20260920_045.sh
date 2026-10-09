#!/bin/bash
set -euo pipefail
cd "${FUSION_ROOT:?set FUSION_ROOT to the package root}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
${PYTHON:-python} scripts/challenge2/check_complete_transfer_review.py \
  --plan results/challenge2/c2_export_045/transfer_review_plan.json \
  --output results/checks/c2_transfer_review_045.json
