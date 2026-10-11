#!/bin/bash
# CPU data preparation for the Challenge-1 (DIII-D) entry: raw download -> base views -> li stores -> all-labelled view ->
# vessel / future-actuator features and views -> scalar targets -> 9-scalar view for the HLVF members.  Every command below is the one recorded in Slurm accounting for the
# artifacts our members were trained on (see PROVENANCE.md).  CPU only (not part of the GPU budget).  Run from anywhere:
#   FUSION_ROOT=/path/to/fusion_c1_final PYTHON=/path/to/python bash prepare.sh [N_SHARDS]
set -euo pipefail
cd "${FUSION_ROOT:?set FUSION_ROOT to the package root}"
PY=${PYTHON:-python}; N=${1:-8}; START=${START_STEP:-1}   # START_STEP=k resumes at step k
export CUDA_VISIBLE_DEVICES=""
log() { echo "=== $(date -u +%FT%TZ) $*"; }
shards() { local script=$1 view=$2 split=$3 out=$4 n=$5; for ((k = 0; k < n; k++)); do $PY -u "$script" "$view" "$split" "$out" "$k/$n" & done; wait; }

if [ "$START" -le 1 ]; then
log "1. download (pinned revision + sha256 from data/raw/download_manifest.json)"
mkdir -p data/raw; [ -f data/raw/download_manifest.json ] || cp provenance/download_manifest.json data/raw/download_manifest.json
$PY -u scripts/download_data.py --workers 12

fi
if [ "$START" -le 2 ]; then
log "2. legacy split + base A view (64 train / 32 val frames per shot) + POD"
$PY -u scripts/prepare_data.py --train-only
FUSION_ROOT="$FUSION_ROOT" PYTHON="$PY" bash scripts/optimization/prepare_full.sh
$PY -u scripts/optimization/prepare_scaled_features.py
$PY -u scripts/optimization/prepare_history_features.py --base-dir data/advanced_v1 --output data/advanced_history_v2 --source-pod data/pod_scaled_v2 --pod-output data/pod_history_v2 --workers 8
$PY -u scripts/optimization/prepare_robust_features.py --data-dir data/advanced_history_v2 --source-pod data/pod_history_v2 --output data/pod_history128_robust_v4

fi
if [ "$START" -le 3 ]; then
log "3. test views and native-frame dev views"
$PY -u scripts/prepare_advanced_data.py --output data/advanced_test_v1 --workers 4 --splits test
$PY -u scripts/optimization/prepare_history_features.py --base-dir data/advanced_test_v1 --output data/advanced_history_test_v2 --splits test --no-pod --workers 1
$PY -u scripts/prepare_advanced_data.py --output data/advanced_val_native_v3 --val-frames 0 --workers 4 --splits val --no-consolidate
$PY -u scripts/optimization/prepare_history_features.py --base-dir data/advanced_val_native_v3 --output data/advanced_history_val_native_v3 --splits val --no-pod --workers 4

fi
if [ "$START" -le 4 ]; then
log "4. li / geometry stores (bigmem, ~1.5 h) and the all-labelled view"
[ -f data/.li_stores_done ] || { FUSION_ROOT="$FUSION_ROOT" PYTHON="$PY" bash scripts/optimization/prepare_li_all_072.sh && touch data/.li_stores_done; }
# the all-labelled view copies the official scorer's reference cache of the dev splits; build it with the scorer's own --perfect mode
mkdir -p results/prep_logs
$PY -u scripts/evaluate.py --data-dir data/advanced_history_v2 --split val --perfect --workers 8 --output results/prep_logs/perfect_val.json
$PY -u scripts/evaluate.py --data-dir data/advanced_history_v2 --split legacy_val --perfect --workers 8 --output results/prep_logs/perfect_legacy_val.json
OMP_NUM_THREADS=8 $PY -u scripts/optimization/prepare_alllab_view.py

fi
if [ "$START" -le 5 ]; then
log "5. vessel-current features (filtered derivatives of the raw 0.05 ms currents) and views"
V=results/analysis/vessel_196
shards $V/build_features.py data/advanced_history_v2 train $V/feats "$N"
$PY -u $V/build_features.py data/advanced_history_v2 val $V/feats_v2view
$PY -u $V/build_features.py data/advanced_history_val_native_v3 val $V/feats
$PY -u $V/build_features.py data/advanced_history_v2 legacy_val $V/feats
shards $V/build_features.py data/advanced_history_test_v2 test $V/feats 4
$PY -u $V/make_vessel_view.py train && $PY -u $V/make_vessel_view.py native && $PY -u $V/make_vessel_view.py test
$PY -u $V/make_lv_view.py
$PY -u $V/make_av_view.py train && $PY -u $V/make_av_view.py native && $PY -u $V/make_av_view.py test
$PY -u $V/make_lav_view.py

fi
if [ "$START" -le 6 ]; then
log "6. future-actuator features (anti-causal filtered derivatives, 2-50 ms) and views"
F=results/analysis/future_201
shards $F/build_features.py data/advanced_history_v2 train $F/feats "$N"
shards $F/build_features.py data/advanced_history_val_native_v3 val $F/feats 4
$PY -u $F/build_features.py data/advanced_history_v2 legacy_val $F/feats 0/1
shards $F/build_features.py data/advanced_history_test_vessel_v1 test $F/feats 4
shards $F/build_features.py data/advanced_history_v2 val $F/feats_v2view 4
$PY -u $F/make_fut_view.py train && $PY -u $F/make_fut_view.py test && $PY -u $F/make_fut_view.py native
$PY -u $F/make_lvf_view.py

fi
if [ "$START" -le 7 ]; then
log "7. scalar targets for the direct axis regressors"
$PY -u results/analysis/scalar_probe_160/prep_targets.py
fi
if [ "$START" -le 8 ]; then
log "8. consistency-scalar targets of the appended dev rows and the 9-scalar all-labelled LVF / LAV views (heads members)"
$PY -u results/analysis/heads_217/dev_targets.py 8
$PY -u results/analysis/heads_217/make_heads_view.py
fi
log "done"
