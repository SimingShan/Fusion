#!/bin/bash
set -euo pipefail
cd "${FUSION_ROOT:?set FUSION_ROOT to the package root}"
task_workers="${SLURM_CPUS_PER_TASK:-4}"
export OMP_NUM_THREADS="$task_workers" OPENBLAS_NUM_THREADS="$task_workers" MKL_NUM_THREADS="$task_workers"
task_source=results/training/c1_sources_040
${PYTHON:-python} -u "$task_source/scripts/train_pod.py" \
  --data-dir data/advanced_history_v2 --pod-dir data/pod_history128_robust_v4 \
  --output results/training/c1_cold_040 --width 256 --depth 3 --epochs 600 \
  --batch-size 2048 --score-every 50 --seconds 18000 --ridge-alpha -1 \
  --workers "$task_workers" --thomson-dropout 0.10 --seed 2080
${PYTHON:-python} -u "$task_source/scripts/train_pod.py" \
  --data-dir data/advanced_history_v2 --pod-dir data/pod_history128_robust_v4 \
  --output results/training/c1_20260920_040 --init results/training/c1_cold_040/best.pt \
  --width 256 --depth 3 --li-dir data/geometry_history_v4 --li-sampled-only \
  --li-weight 0.02 --geometry-weight 0 --frame-weighting none --thomson-dropout 0.10 \
  --gradient-region envelope --epochs 100 --score-every 20 --batch-size 512 \
  --lr 0.0001 --seconds 2400 --workers "$task_workers" --seed 2081
${PYTHON:-python} -u scripts/optimization/finish_c1_candidate.py \
  --training results/training/c1_20260920_040 --candidate 20260920-040 --workers "$task_workers"
