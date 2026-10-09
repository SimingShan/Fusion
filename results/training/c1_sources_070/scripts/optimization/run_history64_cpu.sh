#!/bin/bash
set -euo pipefail
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=""
${PYTHON:-python} -u scripts/optimization/prepare_pod_view.py --data-dir data/advanced_history_v2 --source-pod data/pod_history_v2 --output data/pod_history64_v2 --rank 64 --feature-group all
${PYTHON:-python} -u scripts/train_pod.py --model pod_deeponet --data-dir data/advanced_history_v2 --pod-dir data/pod_history64_v2 --output results/optimization/pod_history64_cpu --width 128 --depth 3 --epochs 200 --batch-size 2048 --score-every 50 --seconds 1500 --ridge-alpha -1 --workers 8
