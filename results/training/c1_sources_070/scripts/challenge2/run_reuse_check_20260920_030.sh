#!/bin/bash
set -euo pipefail
cd "${FUSION_ROOT:?set FUSION_ROOT to the package root}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
${PYTHON:-python} -u scripts/challenge2/check_export_reuse.py \
  --parent results/challenge2/c2_export_016 --run results/challenge2/c2_export_030 \
  --output results/checks/c2_export_reuse_030.json
