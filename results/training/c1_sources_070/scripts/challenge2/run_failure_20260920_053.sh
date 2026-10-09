#!/bin/bash
set -euo pipefail
cd "${FUSION_ROOT:?set FUSION_ROOT to the package root}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
${PYTHON:-python} -u scripts/challenge2/diagnose_native_failure.py \
  --run results/challenge2/c2_export_smoke_050 --index 0 --frame 70 \
  --output results/challenge2/c2_failure_053
