#!/bin/bash
set -euo pipefail
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
${PYTHON:-python} -u scripts/optimization/prepare_robust_features.py --data-dir data/advanced_history_v2 --source-pod data/pod_history_v2 --output data/pod_history128_robust_v4
${PYTHON:-python} -u scripts/train_pod.py --data-dir data/advanced_history_v2 --pod-dir data/pod_history128_robust_v4 --output results/optimization/pod_history128_robust_seed2027_v4 --width 256 --depth 3 --epochs 600 --batch-size 2048 --score-every 50 --seconds 1800 --ridge-alpha -1 --workers 4 --thomson-dropout 0.10 --seed 2027
