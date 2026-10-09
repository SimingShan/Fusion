#!/bin/bash
set -euo pipefail
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
${PYTHON:-python} -u scripts/train_pod.py --data-dir data/advanced_v1 --pod-dir data/pod_scaled_v2 --output results/optimization/pod_scaled_li_v2 --init results/optimization/pod_scaled_v2/best.pt --li-dir data/li_full_v1 --li-sampled-only --li-weight 0.02 --gradient-region envelope --epochs 100 --score-every 20 --batch-size 512 --lr 0.0001 --seconds 1800 --workers 4
