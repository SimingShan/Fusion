#!/bin/bash
# Inference entrypoint: blend the trained members on the DIII-D public test, apply the closure and the gate-aware Z-only axis shift,
# apply the closed-loop elongation and li adjustments, and package the submission.   FUSION_ROOT=/path/to/fusion_c1_final PYTHON=/path/to/python bash predict.sh [NAME]
# Works after train.sh (members/ trained locally) or from the published weights alone (weights/, default WEIGHTS=weights).
set -euo pipefail
cd "${FUSION_ROOT:?set FUSION_ROOT to the package root}"
PY=${PYTHON:-python}; NAME=${1:-final}; C=results/candidates/$NAME; mkdir -p "$C"
W=${WEIGHTS:-weights}; TRAIN=results/training/scale150/src/scripts/train_pod.py; PAR=${PARALLEL:-4}
M=${MEMBERS_DIR:-members}   # optional overrides below; the defaults are the submitted entry
seeds() { local out="" r a n i; for r in ${1//,/ }; do a=${r%%:*}; n=${r##*:}; for ((i = 0; i < n; i++)); do out="$out $((a + i))"; done; done; echo $out; }
LF=${LVF_FAMILY:-HLVF}   # B-family members: HLVF = LVF recipe + consistency-scalar heads (submitted entry); LVF = without heads
if [ "$LF" = "HLVF" ]; then LVF_DEFAULT=6301:24; else LVF_DEFAULT=6001:17,6101:7; fi
LVF_LIST=$(seeds "${LVF_SEEDS:-$LVF_DEFAULT}"); LAV_LIST=$(seeds "${LAV_SEEDS:-7101:26,7201:14}")   # start:count[,start:count]
Z_BETA=${Z_BETA:-0.75}   # Z-shift strength toward the heads/regressor target (0.5 toward the regressors alone without heads)
LI_B=${LI_B:-0.35}   # li adjustment strength (0 = off)
KAPPA_B=${KAPPA_B:-0.5}   # elongation adjustment strength (0 = off)
declare -A TEST_VIEW=([LVF]=data/advanced_history_test_vf_v1 [HLVF]=data/advanced_history_test_vf_v1 [LAV]=data/advanced_test_vessel_v1)

export_member() {   # member test predictions from the local endpoint, else from the published checkpoint
  local fam=$1 seed=$2 d=$M/$1_s$2
  [ -f "$d/test/diii_d_public_test.npz" ] && return 0
  local ck="$d/muon/checkpoints/epoch_0030.pt"; [ -f "$ck" ] || ck="$W/members/$1_s$2/epoch_0030.pt"
  [ -f "$ck" ] || { echo "no checkpoint for $1_s$2" >&2; return 1; }
  mkdir -p "$d/test"; cp "$ck" "$d/test/best.pt"
  $PY "$TRAIN" --data-dir "${TEST_VIEW[$fam]}" --output "$d/test" --predict-only --split test --workers 2 > "$d/test/predict.log" 2>&1
}
jobs=0
for s in $LVF_LIST; do export_member $LF $s & jobs=$((jobs + 1)); [ $((jobs % PAR)) -eq 0 ] && wait; done; wait
for s in $LAV_LIST; do export_member LAV $s & jobs=$((jobs + 1)); [ $((jobs % PAR)) -eq 0 ] && wait; done; wait
# closure files and direct-regressor predictions: from train.sh outputs, else from the published weights
CL=results/candidates/20261004-080/closure
[ -f $CL/closure64.pt ] || { mkdir -p $CL; cp "$W/closure/basis512.npz" "$W/closure/closure64.pt" $CL/; }
[ -f results/analysis/scalar_probe_160/pred_direct_vf60_test.npy ] || $PY -u results/analysis/scalar_probe_160/predict_direct_vf.py "$W/direct_vf60" direct_vf60

S=""
for s in $LVF_LIST; do d=$M/${LF}_s$s/test; [ -f $d/diii_d_public_test.npz ] || { echo "missing $d" >&2; exit 1; }; S="$S B=$d"; done
for s in $LAV_LIST; do d=$M/LAV_s$s/test; [ -f $d/diii_d_public_test.npz ] || { echo "missing $d" >&2; exit 1; }; S="$S A=$d"; done
echo $S | tr ' ' '\n' > "$C/blend_specs.txt"
# equal per-entry weights; B-only on Thomson-poor (gated) frames; POD-512 closure
$PY -u results/training/muon161/blend_from_members.py "$C/diii_d_public_test_closed.npz" $S
# Z-only interior axis shift (beta 0.5) toward the direct regressors, skipped on gated frames
$PY -u results/analysis/scalar_probe_160/make_gated_targets.py direct_vf60 test
HEADS_ENV=""
if [ "$LF" = "HLVF" ]; then
  # targets from the HLVF members' consistency-scalar heads: Z target = (heads + regressors) / 2, kappa target gains the heads as a third vote
  for s in $LVF_LIST; do echo "$M/${LF}_s$s/test"; done > "$C/heads_members.txt"
  $PY -u results/analysis/heads_217/heads_targets.py "$C/heads_members.txt" results/analysis/scalar_probe_160/pred_direct_vf60g_test.npy "$C/heads"
  ZREG="$PWD/$C/heads_zreg.npy"; HEADS_ENV="$PWD/$C/heads_heads.npy"   # absolute: export_warp_z.py resolves REG against its own directory
else
  ZREG=pred_direct_vf60g_test.npy; Z_BETA=${Z_BETA_NOHEADS:-0.5}
fi
WARP_SRC="$C/diii_d_public_test_closed.npz" BAR=0 BAZ=$Z_BETA REG="$ZREG" $PY -u results/analysis/scalar_probe_160/export_warp_z.py "$NAME" "$Z_BETA" 1
J=results/analysis/jkappa_211; LI_TGT="$C/li_target.npy"; LI_FRAC=$LI_B
if [ "$KAPPA_B" != "0" ]; then
  # elongation adjustment: target = mean of (a) the members' own official kappa and (b) the direct regressors' kappa, used only where (a) and
  # (b) move kappa the same way; realised by a radial compression/expansion about the magnetic axis, solved per frame on the official kappa
  # (closed loop, realised value of the float32 field verified); Thomson-poor frames untouched. CPU only.
  { for s in $LVF_LIST; do echo "1 $M/${LF}_s$s/test/diii_d_public_test.npz"; done; for s in $LAV_LIST; do echo "1 $M/LAV_s$s/test/diii_d_public_test.npz"; done; } > "$C/members_test.txt"
  NO_LI=1 $PY -u $J/member_scalars.py "$C/members_test.txt" "$C/member_scalars.npz" "$C/diii_d_public_test.npz" "${KAPPA_WORKERS:-8}"
  HEADS="$HEADS_ENV" AGREE=1 LIB=$LI_B $PY $J/make_targets.py results/analysis/scalar_probe_160/pred_direct_vf60g_test.npy "$C/tgt" "$C/member_scalars.npz"
  KMODE=r $PY -u $J/kappa_adjust3.py "$C/diii_d_public_test.npz" "$C/diii_d_public_test_k.npz" "$C/tgt_kappa.npy" "$KAPPA_B" "${KAPPA_WORKERS:-8}" 1.4
  mv "$C/diii_d_public_test_k.npz" "$C/diii_d_public_test.npz"
  LI_TGT="$C/tgt_li.npy"; LI_FRAC=1.0   # li target anchored on the pre-elongation li: li_base + LI_B (li_direct - li_base), realised exactly
fi
if [ "$LI_B" != "0" ]; then   # boundary-preserving li adjustment toward the direct regressors' li (fraction LI_B; Thomson-poor frames untouched)
  [ "$KAPPA_B" != "0" ] || $PY -c "import numpy as np; g = np.load('results/analysis/scalar_probe_160/pred_direct_vf60g_test.npy'); np.save('$C/li_target.npy', g[:, 6].astype(np.float64))"
  $PY -u results/analysis/lihead_210/li_adjust.py "$C/diii_d_public_test.npz" "$C/diii_d_public_test_li.npz" "$LI_TGT" "$LI_FRAC" - "${LI_WORKERS:-8}"
  mv "$C/diii_d_public_test_li.npz" "$C/diii_d_public_test.npz"
fi
$PY -u scripts/package_submission.py --pred "$C/diii_d_public_test.npz" --output "$C/submission.zip"
sha256sum "$C/submission.zip"
