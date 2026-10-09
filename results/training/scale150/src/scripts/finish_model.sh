#!/usr/bin/env bash
set -euo pipefail
model=${1:?Supply deeponet or fourier_deeponet}
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
${PYTHON:-python} -u scripts/evaluate.py --pred "results/$model/validation_predictions.npz" --output "results/$model/score.json" --workers 4
${PYTHON:-python} -u scripts/train.py --model "$model" --predict-only --split test --batch-size 32
${PYTHON:-python} -u scripts/package_submission.py --pred "results/$model/diii_d_public_test.npz" --output "results/$model/submission.zip"
