#!/bin/bash
# Devel-partition run of the physics input field validation (Challenge 2).
# Usage: sbatch --account=lu_lu --partition=devel --cpus-per-task=4 --mem=32G --time=01:00:00 \
#        -o logs/challenge2/physics_fields_%j.log scripts/challenge2/run_physics_fields_20260921.sh
set -euo pipefail
cd "${FUSION_ROOT:?set FUSION_ROOT to the package root}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export MKL_NUM_THREADS=$OMP_NUM_THREADS
${PYTHON:-python} scripts/challenge2/check_physics_fields.py --n-shots 12 --frames-per-shot 6 --mast-shots 3 \
    --out results/challenge2/physics_fields_v1 "$@"
