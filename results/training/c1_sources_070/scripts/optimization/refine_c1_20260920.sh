#!/bin/bash
set -euo pipefail
cd "${FUSION_ROOT:?set FUSION_ROOT to the package root}"
task_workers="${SLURM_CPUS_PER_TASK:-4}"
export OMP_NUM_THREADS="$task_workers" OPENBLAS_NUM_THREADS="$task_workers" MKL_NUM_THREADS="$task_workers"
${PYTHON:-python} -u scripts/train_pod.py \
  --data-dir data/advanced_history_v2 --pod-dir data/pod_history128_robust_v4 \
  --output results/training/c1_20260920_002 \
  --init results/candidates/20260919-001/checkpoints/b.pt \
  --width 256 --depth 3 --li-dir data/geometry_history_v4 \
  --li-sampled-only --li-weight 0.02 --geometry-weight 0.02 \
  --thomson-dropout 0.1 --gradient-region envelope --epochs 80 \
  --score-every 10 --batch-size 512 --lr 0.00003 \
  --seconds 1800 --workers "$task_workers" --seed 2042
