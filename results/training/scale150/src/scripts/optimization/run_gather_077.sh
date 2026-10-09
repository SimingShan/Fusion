#!/bin/bash
set -euo pipefail
cd "${FUSION_ROOT:?set FUSION_ROOT to the package root}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
${PYTHON:-python} -u scripts/optimization/gather_members_075.py --output results/evaluations/blend_gather_077 --workers "${SLURM_CPUS_PER_TASK:-8}" --min-single 0.9940
