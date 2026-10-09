#!/bin/bash
# Full GPU training of the Challenge-2 flux U-Net on DIII-D + synthetic free-boundary MAST equilibria (v4), followed by
# DIII-D test/dev-fold predictions and zero-shot real-MAST inference with the plausibility report.
# Submit: SYN_DIR=data/c2_synth_mast_v1 sbatch --account=lu_lu --partition=gpu --gres=gpu:1 --cpus-per-task=4 --mem=64G --time=08:00:00 \
#         -J c2_cnn_v4_synth -o logs/challenge2/c2_cnn_v4_synth_%j.log scripts/challenge2/run_c2_cnn_synth_20260922.sh
set -euo pipefail
cd "${FUSION_ROOT:?set FUSION_ROOT to the package root}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export MKL_NUM_THREADS=$OMP_NUM_THREADS
OUT=${C2_OUT:-results/challenge2/c2_cnn_v4_synth}
SYN_DIR=${SYN_DIR:-data/c2_synth_mast_v1}
SYN_W=${SYN_WEIGHT:-0.3}
SECONDS_BUDGET=${C2_SECONDS:-25000}
EPOCHS=${C2_EPOCHS:-40}
GRAD_W=${C2_GRAD_WEIGHT:-0.25}
SCALAR_W=${C2_SCALAR_WEIGHT:-0.1}
mkdir -p "$OUT"
nvidia-smi || true
echo "== feature spec: $(${PYTHON:-python} -c 'from fusion import c2_fields_data as cd; print(cd.feature_spec_hash())')"
echo "== train: DIII-D frames-per-shot 32 + synthetic $SYN_DIR (weight $SYN_W), epochs $EPOCHS, seconds $SECONDS_BUDGET, score every 5 epochs"
${PYTHON:-python} scripts/challenge2/train_c2_cnn.py --output "$OUT" --frames-per-shot 32 --epochs "$EPOCHS" \
    --seconds "$SECONDS_BUDGET" --batch-size 64 --lr 1e-3 --base 32 --score-every 5 --workers "$OMP_NUM_THREADS" \
    --gradient-weight "$GRAD_W" --scalar-weight "$SCALAR_W" --field-norm auto --grad-norm ${C2_GRAD_NORM:-gradient} \
    --seed 2026 --synthetic-dir "$SYN_DIR" --synthetic-weight "$SYN_W" --synthetic-val-frames ${SYN_VAL_FRAMES:-1000} ${C2_EXTRA:-}
echo "== DIII-D public test predictions (official layout, 874 shots)"
${PYTHON:-python} scripts/challenge2/train_c2_cnn.py --output "$OUT" --predict-only --split test \
    --data-dir data/advanced_test_v1 --workers "$OMP_NUM_THREADS"
for HALF in low high; do
  if [[ -f data/c2_cnn_kappa_split/$HALF/manifest.json ]]; then
    echo "== dev fold $HALF-kappa half"
    ${PYTHON:-python} scripts/challenge2/train_c2_cnn.py --output "$OUT" --predict-only --split val \
        --val-dir data/c2_cnn_kappa_split/$HALF --pred-name "val_kappa_${HALF}.npz" --workers "$OMP_NUM_THREADS"
  fi
done
echo "== zero-shot real-MAST inference (1206 shots) + plausibility report (best.pt = DIII-D dev-fold selection)"
${PYTHON:-python} scripts/challenge2/predict_mast_cnn.py --checkpoint "$OUT" --output "$OUT/mast" \
    --report-frames-per-shot 8 --workers "$OMP_NUM_THREADS"
if [[ -f "$OUT/best_synthetic.pt" ]]; then
  echo "== real-MAST inference with best_synthetic.pt (synthetic-MAST selection) for comparison"
  ${PYTHON:-python} scripts/challenge2/predict_mast_cnn.py --checkpoint "$OUT/best_synthetic.pt" --output "$OUT/mast_best_synthetic" \
      --report-frames-per-shot 8 --workers "$OMP_NUM_THREADS"
fi
echo "== done"
