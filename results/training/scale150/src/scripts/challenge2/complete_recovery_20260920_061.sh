#!/bin/bash
set -euo pipefail
cd "${FUSION_ROOT:?set FUSION_ROOT to the package root}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
task_run=results/challenge2/c2_export_061
${PYTHON:-python} scripts/challenge2/review_transfer_recovery.py --plan "$task_run/transfer_review_plan.json" --check-plan
${PYTHON:-python} "$task_run/reuse/source_snapshot/scripts/challenge2/reuse_restart_export.py" adopt --run "$task_run"
${PYTHON:-python} -u "$task_run/source_snapshot/scripts/challenge2/export_mast_predictions.py" --output "$task_run" --workers 4
${PYTHON:-python} "$task_run/reuse/source_snapshot/scripts/challenge2/reuse_restart_export.py" finalize --run "$task_run"
${PYTHON:-python} scripts/challenge2/summarize_export_coverage.py --run "$task_run" --output "$task_run/coverage_review.json"
