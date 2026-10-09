#!/bin/bash
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=""
${PYTHON:-python} -u scripts/prepare_advanced_data.py --output data/advanced_val_native_v3 --val-frames 0 --workers 4 --splits val --no-consolidate
${PYTHON:-python} -u scripts/optimization/prepare_history_features.py --base-dir data/advanced_val_native_v3 --output data/advanced_history_val_native_v3 --splits val --no-pod --workers 4
mkdir -p results/optimization/native_validation_v3/scaled_li results/optimization/native_validation_v3/robust_history
cp results/optimization/pod_scaled_li_v2/best.pt results/optimization/native_validation_v3/scaled_li/best.pt
cp results/optimization/pod_history64_robust_v3/best.pt results/optimization/native_validation_v3/robust_history/best.pt
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
${PYTHON:-python} -u scripts/train_pod.py --data-dir data/advanced_val_native_v3 --output results/optimization/native_validation_v3/scaled_li --predict-only --split val --workers 8 --batch-size 512
${PYTHON:-python} -u scripts/train_pod.py --data-dir data/advanced_history_val_native_v3 --output results/optimization/native_validation_v3/robust_history --predict-only --split val --workers 8 --batch-size 512
${PYTHON:-python} -u scripts/optimization/gated_fourier_blend.py --data-dir data/advanced_val_native_v3 --primary results/optimization/native_validation_v3/scaled_li --expert-name scaled-li-POD-DeepONet --secondary results/optimization/native_validation_v3/robust_history --secondary-name robust-history-POD-DeepONet --output results/optimization/native_validation_v3/combined --workers 8 --splits val --skip-test-coverage
