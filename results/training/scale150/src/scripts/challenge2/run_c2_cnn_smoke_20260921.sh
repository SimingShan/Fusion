#!/bin/bash
# CPU smoke test of the Challenge-2 flux U-Net pipeline (feature-spec v2; defaults: scalar weight 0.1, field loss
# normalised by the training-target variance). Set C2_OUT to keep an earlier smoke directory.
# Usage: sbatch --account=lu_lu --partition=bigmem --cpus-per-task=8 --mem=150G --time=02:00:00 \
#        -o logs/challenge2/c2_cnn_smoke_%j.log scripts/challenge2/run_c2_cnn_smoke_20260921.sh
#   (devel: --partition=devel --cpus-per-task=4 --mem=32G)
set -euo pipefail
cd "${FUSION_ROOT:?set FUSION_ROOT to the package root}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export MKL_NUM_THREADS=$OMP_NUM_THREADS
OUT=${C2_OUT:-results/challenge2/c2_cnn_smoke_20260921}
rm -rf "$OUT"; mkdir -p "$OUT"
echo "== train (60 shots x 16 frames, 3 epochs, score every epoch on 8 dev shots)"
${PYTHON:-python} scripts/challenge2/train_c2_cnn.py --output "$OUT" --max-train-shots 60 --frames-per-shot 16 \
    --epochs 3 --seconds 3600 --batch-size 64 --base 32 --score-every 1 --val-max-shots 8 --workers "$OMP_NUM_THREADS" --seed 2026
echo "== predict-only re-run on the same 8 dev shots (checks best.pt reload path)"
${PYTHON:-python} scripts/challenge2/train_c2_cnn.py --output "$OUT" --predict-only --split val --val-max-shots 8 \
    --pred-name predict_only_val.npz --workers "$OMP_NUM_THREADS"
echo "== MAST inference on the first 5 official shots + plausibility report"
${PYTHON:-python} scripts/challenge2/predict_mast_cnn.py --checkpoint "$OUT" --output "$OUT/mast_smoke" --max-shots 5 \
    --report-frames-per-shot 6 --workers "$OMP_NUM_THREADS"
echo "== done"
