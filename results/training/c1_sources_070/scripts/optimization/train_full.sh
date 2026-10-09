#!/bin/bash
set -euo pipefail
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
${PYTHON:-python} -u scripts/train_pod.py --data-dir data/advanced_v1 --pod-dir data/pod_full_v1 --output results/optimization/pod_full_v1 --oracle-only --batch-size 512
${PYTHON:-python} -u scripts/train_pod.py --data-dir data/advanced_v1 --pod-dir data/pod_full_v1 --output results/optimization/pod_full_v1 --epochs 600 --score-every 50 --batch-size 1024 --seconds 2400 --ridge-alpha -1
${PYTHON:-python} -u scripts/train_pod.py --model pod_fourier_deeponet --data-dir data/advanced_v1 --pod-dir data/pod_full_v1 --output results/optimization/pod_fourier_full_v1 --init results/optimization/pod_full_v1/best.pt --epochs 20 --score-every 5 --batch-size 64 --lr 0.0001 --seconds 2400
${PYTHON:-python} scripts/optimization/plot_progress.py
