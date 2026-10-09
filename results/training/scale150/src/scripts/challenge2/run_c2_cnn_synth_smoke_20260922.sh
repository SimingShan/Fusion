#!/bin/bash
# CPU smoke test of the synthetic-augmented Challenge-2 U-Net training path (feature spec v3 unchanged):
# 60 DIII-D shots x 16 frames + synthetic MAST frames, 3 epochs, scored every epoch on 8 DIII-D dev shots and on the
# held-out synthetic MAST split with the official scorer machinery; then MAST inference on 5 shots.
# Usage: SYN_DIR=data/c2_synth_mast_v1 sbatch --account=lu_lu --partition=bigmem --cpus-per-task=32 --mem=200G --time=02:00:00 \
#        -J c2_synth_smoke -o logs/challenge2/c2_synth_smoke_%j.log scripts/challenge2/run_c2_cnn_synth_smoke_20260922.sh
set -euo pipefail
cd "${FUSION_ROOT:?set FUSION_ROOT to the package root}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export MKL_NUM_THREADS=$OMP_NUM_THREADS
OUT=${C2_OUT:-results/challenge2/c2_synth_smoke_20260922}
SYN_DIR=${SYN_DIR:-data/c2_synth_mast_v1}
SYN_W=${SYN_WEIGHT:-0.3}
rm -rf "$OUT"; mkdir -p "$OUT"
echo "== feature spec: $(${PYTHON:-python} -c 'from fusion import c2_fields_data as cd; print(cd.feature_spec_hash())')"
echo "== train (60 DIII-D shots x 16 frames + synthetic $SYN_DIR at weight $SYN_W, 3 epochs)"
${PYTHON:-python} scripts/challenge2/train_c2_cnn.py --output "$OUT" --max-train-shots 60 --frames-per-shot 16 \
    --epochs 3 --seconds 5400 --batch-size 64 --base 32 --score-every 1 --val-max-shots 8 --workers "$OMP_NUM_THREADS" --seed 2026 \
    --grad-norm gradient --synthetic-dir "$SYN_DIR" --synthetic-weight "$SYN_W" --synthetic-val-frames ${SYN_VAL_FRAMES:-600} ${C2_EXTRA:-}
echo "== predict-only re-run on the same 8 dev shots (checks best.pt reload path)"
${PYTHON:-python} scripts/challenge2/train_c2_cnn.py --output "$OUT" --predict-only --split val --val-max-shots 8 \
    --pred-name predict_only_val.npz --workers "$OMP_NUM_THREADS"
echo "== MAST inference on the first 5 official shots + plausibility report"
${PYTHON:-python} scripts/challenge2/predict_mast_cnn.py --checkpoint "$OUT" --output "$OUT/mast_smoke" --max-shots 5 \
    --report-frames-per-shot 6 --workers "$OMP_NUM_THREADS"
echo "== done"
