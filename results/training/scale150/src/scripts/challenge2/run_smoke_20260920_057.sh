#!/bin/bash
set -euo pipefail
cd "${FUSION_ROOT:?set FUSION_ROOT to the package root}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
task_run=results/challenge2/c2_export_smoke_057
${PYTHON:-python} -u "$task_run/source_snapshot/scripts/challenge2/export_mast_predictions.py" --output "$task_run" --workers 1
${PYTHON:-python} scripts/challenge2/check_native_export_equivalence.py \
  --original results/challenge2/c2_export_smoke_044 --updated "$task_run" \
  --output results/checks/c2_fresh_export_parity_057.json
