#!/bin/bash
set -euo pipefail
cd "${FUSION_ROOT:?set FUSION_ROOT to the package root}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
task_run=results/challenge2/c2_export_061
${PYTHON:-python} scripts/challenge2/reuse_restart_export.py prepare --run "$task_run" \
  --parent results/challenge2/c2_export_045 \
  --parity-original results/challenge2/c2_export_smoke_044 \
  --parity-updated results/challenge2/c2_export_smoke_057 \
  --correction results/challenge2/c2_export_smoke_058
${PYTHON:-python} scripts/challenge2/review_transfer_recovery.py \
  --plan "$task_run/transfer_review_plan.json" --freeze-plan --check-plan
${PYTHON:-python} scripts/challenge2/check_recovery_transfer_review.py \
  --plan "$task_run/transfer_review_plan.json" --output results/checks/c2_transfer_recovery_review_061.json
