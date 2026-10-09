#!/bin/bash
set -euo pipefail
cd "${FUSION_ROOT:?set FUSION_ROOT to the package root}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
${PYTHON:-python} -u scripts/challenge2/check_transfer_review.py \
  --plan results/challenge2/c2_export_030/transfer_review_plan.json \
  --output results/checks/c2_transfer_review_030.json
