#!/bin/bash
set -euo pipefail
cd "${FUSION_ROOT:?set FUSION_ROOT to the package root}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=""
${PYTHON:-python} -u scripts/optimization/benchmark_training_capacity.py \
  --source results/training/c1_sources_056 --output results/checks/c1_capacity_064.json --workers 4
