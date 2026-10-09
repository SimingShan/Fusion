#!/bin/bash
set -euo pipefail
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=""
${PYTHON:-python} -u scripts/optimization/gated_fourier_blend.py --data-dir data/advanced_v1 --primary results/optimization/pod_scaled_li_v2 --expert-name scaled-li-POD-DeepONet --secondary results/optimization/pod_history_geometry_v4 --secondary-name robust-history-geometry-POD-DeepONet --output results/optimization/quality_gate_scaled_geometry_v6 --workers 8 --skip-test-coverage
mkdir -p results/optimization/native_geometry_v6/robust_geometry
cp results/optimization/pod_history_geometry_v4/best.pt results/optimization/native_geometry_v6/robust_geometry/best.pt
${PYTHON:-python} -u scripts/train_pod.py --data-dir data/advanced_history_val_native_v3 --output results/optimization/native_geometry_v6/robust_geometry --predict-only --split val --workers 8 --batch-size 512
${PYTHON:-python} -u scripts/optimization/gated_fourier_blend.py --data-dir data/advanced_val_native_v3 --primary results/optimization/native_validation_v3/scaled_li --expert-name scaled-li-POD-DeepONet --secondary results/optimization/native_geometry_v6/robust_geometry --secondary-name robust-history-geometry-POD-DeepONet --output results/optimization/native_geometry_v6/combined --workers 8 --splits val --skip-test-coverage
