#!/bin/bash
set -euo pipefail
task_workers="${SLURM_CPUS_PER_TASK:-2}"
export OMP_NUM_THREADS="$task_workers" OPENBLAS_NUM_THREADS="$task_workers" MKL_NUM_THREADS="$task_workers"
export CUDA_VISIBLE_DEVICES=""
${PYTHON:-python} -u scripts/optimization/gated_fourier_blend.py --data-dir data/advanced_v1 --primary results/optimization/pod_scaled_li_v2 --expert-name scaled-li-POD-DeepONet --secondary results/optimization/pod_history128_li_v6 --secondary-name robust-history128-li-POD-DeepONet --output results/optimization/quality_gate_scaled_history128_li_v8 --workers "$task_workers" --skip-test-coverage
mkdir -p results/optimization/native_history128_li_v8/robust_history128_li
cp results/optimization/pod_history128_li_v6/best.pt results/optimization/native_history128_li_v8/robust_history128_li/best.pt
${PYTHON:-python} -u scripts/train_pod.py --data-dir data/advanced_history_val_native_v3 --output results/optimization/native_history128_li_v8/robust_history128_li --predict-only --split val --workers "$task_workers" --batch-size 512
${PYTHON:-python} -u scripts/optimization/gated_fourier_blend.py --data-dir data/advanced_val_native_v3 --primary results/optimization/native_validation_v3/scaled_li --expert-name scaled-li-POD-DeepONet --secondary results/optimization/native_history128_li_v8/robust_history128_li --secondary-name robust-history128-li-POD-DeepONet --output results/optimization/native_history128_li_v8/combined --workers "$task_workers" --splits val --skip-test-coverage
