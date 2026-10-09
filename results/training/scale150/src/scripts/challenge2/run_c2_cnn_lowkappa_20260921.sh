#!/bin/bash
# Kappa extrapolation stress test: same recipe as run_c2_cnn_20260921.sh but trained only on the LOWEST 60%
# of DIII-D train shots by label elongation (results/challenge2/c2_cnn_kappa, from run_c2_kappa_labels_20260921.sh),
# then scored on the low/high-kappa halves of the native dev fold.
# Submit: sbatch --account=lu_lu --partition=gpu --gres=gpu:1 --cpus-per-task=4 --mem=64G --time=08:00:00 \
#         --dependency=afterok:<kappa job> -J c2_cnn_lowk -o logs/challenge2/c2_cnn_lowk_%j.log \
#         scripts/challenge2/run_c2_cnn_lowkappa_20260921.sh
set -euo pipefail
cd "${FUSION_ROOT:?set FUSION_ROOT to the package root}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export MKL_NUM_THREADS=$OMP_NUM_THREADS
OUT=results/challenge2/c2_cnn_lowkappa_20260921
LIST=results/challenge2/c2_cnn_kappa/train_low_kappa_60_shots.json
test -f "$LIST" || { echo "missing $LIST (run run_c2_kappa_labels_20260921.sh first)"; exit 1; }
mkdir -p "$OUT"
nvidia-smi || true
${PYTHON:-python} scripts/challenge2/train_c2_cnn.py --output "$OUT" --frames-per-shot 32 --epochs 40 --seconds 22000 \
    --batch-size 64 --lr 1e-3 --base 32 --score-every 5 --workers "$OMP_NUM_THREADS" --seed 2026 --train-shot-list "$LIST"
for HALF in low high; do
  ${PYTHON:-python} scripts/challenge2/train_c2_cnn.py --output "$OUT" --predict-only --split val \
      --val-dir data/c2_cnn_kappa_split/$HALF --pred-name "val_kappa_${HALF}.npz" --workers "$OMP_NUM_THREADS"
done
${PYTHON:-python} scripts/challenge2/predict_mast_cnn.py --checkpoint "$OUT" --output "$OUT/mast" --max-shots 200 \
    --report-frames-per-shot 6 --workers "$OMP_NUM_THREADS"
echo "== done"
