#!/bin/bash
set -euo pipefail
cd "${FUSION_ROOT:?set FUSION_ROOT to the package root}"
task_workers="${SLURM_CPUS_PER_TASK:-4}"
export OMP_NUM_THREADS="$task_workers" OPENBLAS_NUM_THREADS="$task_workers" MKL_NUM_THREADS="$task_workers" CUDA_VISIBLE_DEVICES=""
task_source=results/training/c1_sources_066
${PYTHON:-python} -u "$task_source/scripts/train_pod.py" \
  --data-dir data/advanced_history_v2 --pod-dir data/pod_history128_robust_v4 \
  --output results/training/c1_cold_066 --width 512 --depth 4 --epochs 600 \
  --batch-size 2048 --score-every 50 --seconds 15000 --ridge-alpha -1 \
  --workers "$task_workers" --thomson-dropout 0.10 --seed 2086
${PYTHON:-python} -u "$task_source/scripts/train_pod.py" \
  --data-dir data/advanced_history_v2 --pod-dir data/pod_history128_robust_v4 \
  --output results/training/c1_20260920_066 --init results/training/c1_cold_066/best.pt \
  --width 512 --depth 4 --li-dir data/geometry_history_v4 --li-sampled-only \
  --li-weight 0.02 --geometry-weight 0 --frame-weighting none --thomson-dropout 0.10 \
  --gradient-region envelope --epochs 100 --score-every 20 --batch-size 512 \
  --lr 0.0001 --seconds 3600 --workers "$task_workers" --seed 2087
${PYTHON:-python} -u scripts/optimization/finish_c1_candidate.py \
  --training results/training/c1_20260920_066 --candidate 20260920-066 --workers "$task_workers"
