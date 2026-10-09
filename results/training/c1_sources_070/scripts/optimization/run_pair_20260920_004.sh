#!/bin/bash
set -euo pipefail
cd "${FUSION_ROOT:?set FUSION_ROOT to the package root}"
task_workers="${SLURM_CPUS_PER_TASK:-4}"
export OMP_NUM_THREADS="$task_workers" OPENBLAS_NUM_THREADS="$task_workers" MKL_NUM_THREADS="$task_workers"
${PYTHON:-python} -u scripts/optimization/train_pair_policy.py \
  --output results/training/c1_20260920_004 --epochs 60 --seconds 2400 \
  --score-every 5 --workers "$task_workers"
