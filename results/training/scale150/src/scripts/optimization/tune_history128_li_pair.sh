#!/bin/bash
set -euo pipefail
task_workers="${SLURM_CPUS_PER_TASK:-2}"
export OMP_NUM_THREADS="$task_workers" OPENBLAS_NUM_THREADS="$task_workers" MKL_NUM_THREADS="$task_workers"
${PYTHON:-python} -u scripts/optimization/tune_pair_blend.py --data-dir data/advanced_v1 --primary results/optimization/pod_scaled_li_v2 --secondary results/optimization/pod_history128_li_v6 --output results/optimization/quality_gate_history128_li_tuned_v9 --primary-weights 0.25 0.5 --existing-075-summary results/optimization/quality_gate_scaled_history128_li_v8/summary.json --workers "$task_workers"
${PYTHON:-python} -u scripts/optimization/tune_pair_blend.py --data-dir data/advanced_val_native_v3 --primary results/optimization/native_validation_v3/scaled_li --secondary results/optimization/native_history128_li_v8/robust_history128_li --output results/optimization/native_history128_li_tuned_v9 --primary-weights 0.25 0.5 --splits val --existing-075-summary results/optimization/native_history128_li_v8/combined/summary.json --workers "$task_workers"
