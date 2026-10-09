#!/bin/bash
# Submit with: sbatch --account=lu_lu --partition=devel --cpus-per-task=8 --mem=32G --time=00:30:00 scripts/optimization/prepare_native_training.sh
set -euo pipefail
cd "${FUSION_ROOT:?set FUSION_ROOT to the package root}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=""
${PYTHON:-python} -u scripts/optimization/prepare_native_training.py --workers "${SLURM_CPUS_PER_TASK:-8}" "$@"
