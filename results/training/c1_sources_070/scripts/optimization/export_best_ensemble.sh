#!/bin/bash
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=""
${PYTHON:-python} -u scripts/optimization/export_gated_ensemble.py --workers 1 --device cpu
${PYTHON:-python} -u scripts/package_submission.py --pred results/optimization/submission_scaled_robust_v5/diii_d_public_test.npz --output results/optimization/submission_scaled_robust_v5/submission.zip
