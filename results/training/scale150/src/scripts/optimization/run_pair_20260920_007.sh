#!/bin/bash
set -euo pipefail
cd "${FUSION_ROOT:?set FUSION_ROOT to the package root}"
task_workers="${SLURM_CPUS_PER_TASK:-4}"
export OMP_NUM_THREADS="$task_workers" OPENBLAS_NUM_THREADS="$task_workers" MKL_NUM_THREADS="$task_workers"
${PYTHON:-python} -u scripts/optimization/train_pair_policy.py \
  --output results/training/c1_20260920_007 --epochs 40 --seconds 1500 \
  --lr 0.000005 --thomson-dropout 0 --geometry-weight 0 --li-weight 0.01 \
  --score-every 5 --workers "$task_workers" --seed 2057
${PYTHON:-python} -u scripts/optimization/finish_c1_candidate.py \
  --training results/training/c1_20260920_007 --candidate 20260920-007 --workers "$task_workers"
