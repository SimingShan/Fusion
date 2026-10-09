#!/bin/bash
set -euo pipefail
cd "${FUSION_ROOT:?set FUSION_ROOT to the package root}"
task_workers="${SLURM_CPUS_PER_TASK:-4}"
export OMP_NUM_THREADS="$task_workers" OPENBLAS_NUM_THREADS="$task_workers" MKL_NUM_THREADS="$task_workers"
${PYTHON:-python} scripts/optimization/check_exact_axis.py --output results/checks/c1_024_axis.json
${PYTHON:-python} -u scripts/optimization/train_pair_policy.py \
  --output results/training/c1_20260920_024 --epochs 40 --seconds 2400 --lr 0.000005 \
  --thomson-dropout 0 --geometry-weight 0 --exact-axis-weight 0.03 --li-weight 0.01 \
  --score-every 5 --workers "$task_workers" --seed 2057
${PYTHON:-python} -u scripts/optimization/finish_c1_candidate.py \
  --training results/training/c1_20260920_024 --candidate 20260920-024 --workers "$task_workers"
