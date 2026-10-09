#!/bin/bash
set -euo pipefail
cd "${FUSION_ROOT:?set FUSION_ROOT to the package root}"
task_workers="${SLURM_CPUS_PER_TASK:-8}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
${PYTHON:-python} -u scripts/optimization/blend_members.py \
  --members A=results/optimization/native_validation_v3/scaled_li/val_predictions.npz \
            B=results/optimization/native_history128_li_v8/robust_history128_li/val_predictions.npz \
            S040=results/evaluations/20260920-040/native/val_predictions.npz \
            S066=results/evaluations/20260920-066/native/val_predictions.npz \
            X067=results/evaluations/20260920-067/native/val_predictions.npz \
            N002=results/evaluations/20260920-002/native/val_predictions.npz \
            V049=results/evaluations/20260920-049/native/val_predictions.npz \
  --blends A,B A,B,S040 A,B,S040,S066 A,B,S040,S066,X067 A,B,S040,S066,X067,N002 A,B,S040,S066,X067,N002,V049 A,X067,S040,S066 A,N002,S040,S066 \
  --weights-fit --workers "$task_workers" --output results/evaluations/blend_explore_070
