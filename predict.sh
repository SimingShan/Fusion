#!/bin/bash
# Inference entrypoint: blend the trained members on the DIII-D public test, apply the closure and the gate-aware Z-only axis shift,
# and package the submission.   FUSION_ROOT=/path/to/fusion_c1_final PYTHON=/path/to/python bash predict.sh [N_LVF] [N_LAV] [NAME]
# Works after train.sh (members/ trained locally) or from the published weights alone (weights/, default WEIGHTS=weights).
set -euo pipefail
cd "${FUSION_ROOT:?set FUSION_ROOT to the package root}"
PY=${PYTHON:-python}; NLVF=${1:-17}; NLAV=${2:-26}; NAME=${3:-final}; C=results/candidates/$NAME; mkdir -p "$C"
W=${WEIGHTS:-weights}; TRAIN=results/training/scale150/src/scripts/train_pod.py; PAR=${PARALLEL:-4}
declare -A TEST_VIEW=([LVF]=data/advanced_history_test_vf_v1 [LAV]=data/advanced_test_vessel_v1)

export_member() {   # member test predictions from the local endpoint, else from the published checkpoint
  local fam=$1 seed=$2 d=members/$1_s$2
  [ -f "$d/test/diii_d_public_test.npz" ] && return 0
  local ck="$d/muon/checkpoints/epoch_0030.pt"; [ -f "$ck" ] || ck="$W/members/$1_s$2/epoch_0030.pt"
  [ -f "$ck" ] || { echo "no checkpoint for $1_s$2" >&2; return 1; }
  mkdir -p "$d/test"; cp "$ck" "$d/test/best.pt"
  $PY "$TRAIN" --data-dir "${TEST_VIEW[$fam]}" --output "$d/test" --predict-only --split test --workers 2 > "$d/test/predict.log" 2>&1
}
jobs=0
for ((i = 0; i < NLVF; i++)); do export_member LVF $((6001 + i)) & jobs=$((jobs + 1)); [ $((jobs % PAR)) -eq 0 ] && wait; done; wait
for ((i = 0; i < NLAV; i++)); do export_member LAV $((7101 + i)) & jobs=$((jobs + 1)); [ $((jobs % PAR)) -eq 0 ] && wait; done; wait
# closure files and direct-regressor predictions: from train.sh outputs, else from the published weights
CL=results/candidates/20261004-080/closure
[ -f $CL/closure64.pt ] || { mkdir -p $CL; cp "$W/closure/basis512.npz" "$W/closure/closure64.pt" $CL/; }
[ -f results/analysis/scalar_probe_160/pred_direct_vf60_test.npy ] || $PY -u results/analysis/scalar_probe_160/predict_direct_vf.py "$W/direct_vf60" direct_vf60

S=""
for ((i = 0; i < NLVF; i++)); do d=members/LVF_s$((6001 + i))/test; [ -f $d/diii_d_public_test.npz ] || { echo "missing $d" >&2; exit 1; }; S="$S B=$d"; done
for ((i = 0; i < NLAV; i++)); do d=members/LAV_s$((7101 + i))/test; [ -f $d/diii_d_public_test.npz ] || { echo "missing $d" >&2; exit 1; }; S="$S A=$d"; done
echo $S | tr ' ' '\n' > "$C/blend_specs.txt"
# equal per-entry weights; B-only on Thomson-poor (gated) frames; POD-512 closure
$PY -u results/training/muon161/blend_from_members.py "$C/diii_d_public_test_closed.npz" $S
# Z-only interior axis shift (beta 0.5) toward the direct regressors, skipped on gated frames
$PY -u results/analysis/scalar_probe_160/make_gated_targets.py direct_vf60 test
WARP_SRC="$C/diii_d_public_test_closed.npz" BAR=0 BAZ=0.5 REG=pred_direct_vf60g_test.npy $PY -u results/analysis/scalar_probe_160/export_warp_z.py "$NAME" 0.5 1
$PY -u scripts/package_submission.py --pred "$C/diii_d_public_test.npz" --output "$C/submission.zip"
sha256sum "$C/submission.zip"
