#!/bin/bash
set -euo pipefail
cd "${FUSION_ROOT:?set FUSION_ROOT to the package root}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
${PYTHON:-python} -u results/challenge2/c2_export_030/reuse/reuse_valid_export.py \
  --run results/challenge2/c2_export_030
${PYTHON:-python} -u results/challenge2/c2_export_030/source_snapshot/scripts/challenge2/export_mast_predictions.py \
  --output results/challenge2/c2_export_030 --workers 4
${PYTHON:-python} -u results/challenge2/c2_export_030/review_snapshot/scripts/challenge2/finalize_export_reuse.py \
  --run results/challenge2/c2_export_030
