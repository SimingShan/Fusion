#!/bin/bash
set -euo pipefail
cd "${FUSION_ROOT:?set FUSION_ROOT to the package root}"
task_workers="${SLURM_CPUS_PER_TASK:-2}"
export OMP_NUM_THREADS="$task_workers" OPENBLAS_NUM_THREADS="$task_workers" MKL_NUM_THREADS="$task_workers"
${PYTHON:-python} -u scripts/optimization/average_late_checkpoints.py \
  --training results/optimization/pod_history128_li_v6 --output results/training/c1_20260920_049 --tail-count 3
${PYTHON:-python} -u scripts/train_pod.py \
  --data-dir data/advanced_history_v2 --output results/training/c1_20260920_049 \
  --predict-only --split legacy_val --workers "$task_workers"
${PYTHON:-python} -u scripts/optimization/finish_c1_candidate.py \
  --training results/training/c1_20260920_049 --candidate 20260920-049 --workers "$task_workers"
