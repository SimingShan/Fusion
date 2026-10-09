#!/bin/bash
set -euo pipefail
cd "${FUSION_ROOT:?set FUSION_ROOT to the package root}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
${PYTHON:-python} -u scripts/optimization/consistency_probe.py \
  --output results/evaluations/c1_probe_019 --workers "${SLURM_CPUS_PER_TASK:-4}"
