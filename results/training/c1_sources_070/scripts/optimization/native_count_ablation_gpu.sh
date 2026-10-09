#!/bin/bash
set -euo pipefail
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
mkdir -p results/optimization/native_count_ablation_v5
cp results/optimization/pod_history128_robust_seed2027_v4/best.pt results/optimization/native_count_ablation_v5/init.pt
${PYTHON:-python} -u scripts/train_pod.py --data-dir data/advanced_history_v2 --pod-dir data/pod_history128_robust_v4 --output results/optimization/pod_history128_count_control_v5 --init results/optimization/native_count_ablation_v5/init.pt --width 256 --depth 3 --epochs 100 --batch-size 2048 --score-every 25 --seconds 1200 --lr 0.0001 --workers 4 --thomson-dropout 0.10 --seed 2028 --frame-weighting none
${PYTHON:-python} -u scripts/train_pod.py --data-dir data/advanced_history_v2 --pod-dir data/pod_history128_robust_v4 --output results/optimization/pod_history128_count_weighted_v5 --init results/optimization/native_count_ablation_v5/init.pt --width 256 --depth 3 --epochs 100 --batch-size 2048 --score-every 25 --seconds 1200 --lr 0.0001 --workers 4 --thomson-dropout 0.10 --seed 2028 --frame-weighting native_count
