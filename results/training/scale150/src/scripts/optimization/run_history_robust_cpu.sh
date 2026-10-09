#!/bin/bash
set -euo pipefail
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=""
${PYTHON:-python} -u scripts/optimization/prepare_robust_features.py
${PYTHON:-python} -u scripts/train_pod.py --data-dir data/advanced_history_v2 --pod-dir data/pod_history64_robust_v3 --output results/optimization/pod_history64_robust_v3 --width 128 --depth 3 --epochs 300 --batch-size 2048 --score-every 50 --seconds 1500 --ridge-alpha -1 --workers 8 --thomson-dropout 0.10
