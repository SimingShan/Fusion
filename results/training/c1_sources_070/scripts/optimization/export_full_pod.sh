#!/bin/bash
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
${PYTHON:-python} -u scripts/train_pod.py --predict-only --data-dir data/advanced_test_v1 --output results/optimization/pod_full_v1 --split test --batch-size 512 --workers 1
${PYTHON:-python} -u scripts/package_submission.py --pred results/optimization/pod_full_v1/diii_d_public_test.npz --output results/optimization/pod_full_v1/submission.zip
