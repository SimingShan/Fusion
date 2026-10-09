#!/bin/bash
set -euo pipefail
cd "${FUSION_ROOT:?set FUSION_ROOT to the package root}"
task_workers="${SLURM_CPUS_PER_TASK:-4}"
export OMP_NUM_THREADS="$task_workers" OPENBLAS_NUM_THREADS="$task_workers" MKL_NUM_THREADS="$task_workers"
${PYTHON:-python} -u scripts/optimization/initialize_derived_inputs.py \
  --output results/training/c1_init_020 --workers "$task_workers"
${PYTHON:-python} -u scripts/optimization/train_pair_policy.py \
  --secondary results/training/c1_init_020/best.pt --output results/training/c1_20260920_020 \
  --derived-inputs-only --epochs 40 --seconds 2400 --lr 0.0001 --thomson-dropout 0.1 \
  --geometry-weight 0 --li-weight 0.01 --score-every 5 --workers "$task_workers" --seed 2070
${PYTHON:-python} -u scripts/optimization/finish_c1_candidate.py \
  --training results/training/c1_20260920_020 --candidate 20260920-020 --workers "$task_workers"
