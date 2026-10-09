#!/bin/bash
set -euo pipefail
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
${PYTHON:-python} -u scripts/train_pod.py --data-dir data/processed --pod-dir data/pod_legacy --output results/optimization/pod_legacy_fluxloss --epochs 300 --score-every 100 --batch-size 512 --seconds 600 --coefficient-power 2
${PYTHON:-python} -u scripts/train_pod.py --data-dir data/processed --pod-dir data/pod_legacy --output results/optimization/pod_legacy_noridge --epochs 300 --score-every 100 --batch-size 512 --seconds 600 --coefficient-power 1 --ridge-alpha -1
${PYTHON:-python} -u scripts/train_pod.py --data-dir data/processed --pod-dir data/pod_legacy --output results/optimization/pod_legacy_flux_noridge --epochs 300 --score-every 100 --batch-size 512 --seconds 600 --coefficient-power 2 --ridge-alpha -1
${PYTHON:-python} scripts/optimization/plot_progress.py
