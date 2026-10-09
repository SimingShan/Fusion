#!/bin/bash
set -euo pipefail
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=""
${PYTHON:-python} -u scripts/optimization/prepare_pod_view.py --data-dir data/advanced_v1 --source-pod data/pod_full_v1 --output data/pod_currents64_v1 --rank 64 --feature-group currents
${PYTHON:-python} -u scripts/train_pod.py --model pod_deeponet --data-dir data/advanced_v1 --pod-dir data/pod_currents64_v1 --output results/optimization/pod_currents64_cpu --width 128 --depth 2 --epochs 150 --batch-size 2048 --score-every 50 --seconds 1500 --ridge-alpha -1 --workers 8
