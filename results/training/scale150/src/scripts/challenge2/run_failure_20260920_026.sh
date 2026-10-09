#!/bin/bash
set -euo pipefail
cd "${FUSION_ROOT:?set FUSION_ROOT to the package root}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
${PYTHON:-python} -u results/challenge2/c2_sources_026/scripts/challenge2/diagnose_pressure_observations.py \
  --base-run results/challenge2/c2_export_016 --index 175 \
  --reference-failure results/challenge2/c2_failure_025 --output results/challenge2/c2_failure_026
