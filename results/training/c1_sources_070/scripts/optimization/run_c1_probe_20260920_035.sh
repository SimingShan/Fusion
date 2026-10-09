#!/bin/bash
set -euo pipefail
cd "${FUSION_ROOT:?set FUSION_ROOT to the package root}"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
${PYTHON:-python} -u scripts/optimization/nonlinear_consistency_probe.py \
  --source results/evaluations/c1_probe_019 --output results/evaluations/c1_probe_035 --workers 2
