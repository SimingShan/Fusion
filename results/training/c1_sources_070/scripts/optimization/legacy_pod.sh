#!/bin/bash
set -euo pipefail
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
${PYTHON:-python} -u scripts/train_pod.py --data-dir data/processed --pod-dir data/pod_legacy --output results/optimization/pod_legacy --oracle-only
${PYTHON:-python} -u scripts/train_pod.py --data-dir data/processed --pod-dir data/pod_legacy --output results/optimization/pod_legacy --epochs 300 --score-every 50 --batch-size 512 --seconds 1500
${PYTHON:-python} -u scripts/train_pod.py --model pod_fourier_deeponet --data-dir data/processed --pod-dir data/pod_legacy --output results/optimization/pod_fourier_legacy --init results/optimization/pod_legacy/best.pt --epochs 60 --score-every 20 --batch-size 64 --lr 0.0001 --seconds 1500
