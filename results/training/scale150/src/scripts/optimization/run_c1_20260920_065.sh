#!/bin/bash
set -euo pipefail
cd "${FUSION_ROOT:?set FUSION_ROOT to the package root}"
task_workers="${SLURM_CPUS_PER_TASK:-4}"
export OMP_NUM_THREADS="$task_workers" OPENBLAS_NUM_THREADS="$task_workers" MKL_NUM_THREADS="$task_workers" CUDA_VISIBLE_DEVICES=""
task_source=results/training/c1_sources_065
${PYTHON:-python} -u "$task_source/scripts/optimization/expand_checkpoint.py" \
  --checkpoint results/candidates/20260919-001/checkpoints/b.pt \
  --output results/training/c1_initial_065 --pod-dir data/pod_history128_robust_v4 \
  --width 512 --depth 4 --noise-fraction 0.01 --seed 2091 --workers "$task_workers"
${PYTHON:-python} -u "$task_source/scripts/train_pod.py" \
  --data-dir data/advanced_history_v2 --pod-dir data/pod_history128_robust_v4 \
  --output results/training/c1_20260920_065 --init results/training/c1_initial_065/initial.pt \
  --width 512 --depth 4 --li-dir data/geometry_history_v4 --li-weight 0.02 \
  --geometry-weight 0 --frame-weighting none --thomson-dropout 0.10 --gradient-region envelope \
  --epochs 100 --score-every 20 --batch-size 1024 --lr 0.00003 --seconds 5400 \
  --workers "$task_workers" --seed 2092
${PYTHON:-python} -u scripts/optimization/finish_c1_candidate.py \
  --training results/training/c1_20260920_065 --candidate 20260920-065 --workers "$task_workers"
