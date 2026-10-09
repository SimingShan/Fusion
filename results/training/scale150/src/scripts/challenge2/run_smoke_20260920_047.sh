#!/bin/bash
set -euo pipefail
cd "${FUSION_ROOT:?set FUSION_ROOT to the package root}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
task_trial=$((47 + SLURM_ARRAY_TASK_ID))
printf -v task_run 'results/challenge2/c2_export_smoke_%03d' "$task_trial"
${PYTHON:-python} -u "$task_run/source_snapshot/scripts/challenge2/export_mast_predictions.py" \
  --output "$task_run" --workers 1
