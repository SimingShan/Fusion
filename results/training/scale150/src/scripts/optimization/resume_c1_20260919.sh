#!/bin/bash
set -euo pipefail
cd "${FUSION_ROOT:?set FUSION_ROOT to the package root}"
task_workers="${SLURM_CPUS_PER_TASK:-8}"
task_seconds="${1:-10800}"
export OMP_NUM_THREADS="$task_workers" OPENBLAS_NUM_THREADS="$task_workers" MKL_NUM_THREADS="$task_workers"
${PYTHON:-python} -u scripts/train_pod.py \
  --data-dir data/advanced_history_train_native_v7 \
  --pod-dir data/pod_history128_train_native_v7 \
  --output results/training/c1_20260919_001 \
  --init results/candidates/20260919-001/checkpoints/b.pt \
  --width 256 --depth 3 --li-dir data/targets_native_20260919 \
  --li-weight 0.02 --geometry-weight 0 --frame-weighting none \
  --thomson-dropout 0.10 --gradient-region envelope \
  --epochs 80 --score-every 5 --batch-size 4096 --lr 0.00002 \
  --seconds "$task_seconds" --workers "$task_workers" --seed 2030
