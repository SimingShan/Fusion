#!/bin/bash
set -euo pipefail
cd "${FUSION_ROOT:?set FUSION_ROOT to the package root}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
${PYTHON:-python} -u scripts/challenge2/diagnose_export_failure.py \
  --run results/challenge2/c2_export_016 --index 175 --frame 0 \
  --output results/challenge2/c2_failure_025
