#!/bin/bash
# Full GPU training of the machine-agnostic Challenge-2 flux U-Net (DIII-D only) + DIII-D test/dev-fold
# predictions + zero-shot MAST inference with plausibility report.
# Submit: sbatch --account=lu_lu --partition=gpu --gres=gpu:1 --cpus-per-task=4 --mem=64G --time=08:00:00 \
#         -J c2_cnn_main -o logs/challenge2/c2_cnn_main_%j.log scripts/challenge2/run_c2_cnn_20260921.sh
set -euo pipefail
cd "${FUSION_ROOT:?set FUSION_ROOT to the package root}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export MKL_NUM_THREADS=$OMP_NUM_THREADS
OUT=${C2_OUT:-results/challenge2/c2_cnn_20260921}
SECONDS_BUDGET=${C2_SECONDS:-25000}
EPOCHS=${C2_EPOCHS:-40}
SHOT_LIST_ARGS=()
if [[ -n "${C2_SHOT_LIST:-}" ]]; then SHOT_LIST_ARGS=(--train-shot-list "$C2_SHOT_LIST"); fi
mkdir -p "$OUT"
nvidia-smi || true
# Feature-spec v2 (2026-09-21 review fixes): field FiLM conditioning = FIELD_COND_NAMES only, Bt/pressure scalars reach the
# q95/betaN heads only, grid-normalised coordinates, variance-normalised field loss. Loss weights are explicit below.
GRAD_W=${C2_GRAD_WEIGHT:-0.25}
SCALAR_W=${C2_SCALAR_WEIGHT:-0.1}
echo "== feature spec: $(${PYTHON:-python} -c 'from fusion import c2_fields_data as cd; print(cd.feature_spec_hash())')"
echo "== train: frames-per-shot 32, epochs $EPOCHS, seconds $SECONDS_BUDGET, score every 5 epochs on the full native dev fold"
${PYTHON:-python} scripts/challenge2/train_c2_cnn.py --output "$OUT" --frames-per-shot 32 --epochs "$EPOCHS" \
    --seconds "$SECONDS_BUDGET" --batch-size 64 --lr 1e-3 --base 32 --score-every 5 --workers "$OMP_NUM_THREADS" \
    --gradient-weight "$GRAD_W" --scalar-weight "$SCALAR_W" --field-norm auto --grad-norm ${C2_GRAD_NORM:-gradient} \
    --seed 2026 "${SHOT_LIST_ARGS[@]}"
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
echo "== zero-shot MAST inference (1206 shots) + plausibility report"
${PYTHON:-python} scripts/challenge2/predict_mast_cnn.py --checkpoint "$OUT" --output "$OUT/mast" \
    --report-frames-per-shot 8 --workers "$OMP_NUM_THREADS"
echo "== done"
