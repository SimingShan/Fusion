#!/bin/bash
set -euo pipefail
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
${PYTHON:-python} -u scripts/optimization/prepare_geometry_targets.py --data-dir data/advanced_history_v2 --source-li data/li_full_v1 --output data/geometry_history_v4
${PYTHON:-python} -u scripts/train_pod.py --data-dir data/advanced_history_v2 --pod-dir data/pod_history64_robust_v3 --output results/optimization/pod_history_geometry_v4 --init results/optimization/pod_history64_robust_v3/best.pt --width 128 --depth 3 --li-dir data/geometry_history_v4 --li-sampled-only --li-weight 0.02 --geometry-weight 0.02 --thomson-dropout 0.1 --gradient-region envelope --epochs 100 --score-every 20 --batch-size 512 --lr 0.0001 --seconds 1800 --workers 4
