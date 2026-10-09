#!/bin/bash
set -euo pipefail
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
${PYTHON:-python} -u scripts/train_pod.py --data-dir data/advanced_history_v2 --pod-dir data/pod_history128_robust_v4 --output results/optimization/pod_history128_li_v6 --init results/optimization/pod_history128_robust_seed2027_v4/best.pt --width 256 --depth 3 --li-dir data/geometry_history_v4 --li-sampled-only --li-weight 0.02 --geometry-weight 0.0 --frame-weighting none --thomson-dropout 0.10 --gradient-region envelope --epochs 100 --score-every 20 --batch-size 512 --lr 0.0001 --seconds 1800 --workers 4 --seed 2029
