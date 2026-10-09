#!/usr/bin/env bash
# Run on a GPU compute node from the repository root after data download.
set -euo pipefail
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
${PYTHON:-python} scripts/prepare_data.py
${PYTHON:-python} scripts/check_pipeline.py
${PYTHON:-python} -u scripts/train.py --model deeponet --batch-size 256 --epochs 150 --seconds 2400
${PYTHON:-python} -u scripts/train.py --model fourier_deeponet --batch-size 32 --epochs 100 --seconds 2400
for model in deeponet fourier_deeponet; do
    ${PYTHON:-python} scripts/evaluate.py --pred "results/$model/validation_predictions.npz" --output "results/$model/score.json" --workers 4
    ${PYTHON:-python} scripts/train.py --model "$model" --predict-only --split test --batch-size 32
    ${PYTHON:-python} scripts/package_submission.py --pred "results/$model/diii_d_public_test.npz" --output "results/$model/submission.zip"
done
${PYTHON:-python} scripts/plot_results.py
${PYTHON:-python} scripts/report_results.py
