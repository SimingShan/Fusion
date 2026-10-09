#!/bin/bash
# GPU training entrypoint (from scratch; the only GPU-budgeted part).  Requires prepare.sh to have run.
#   FUSION_ROOT=/path/to/fusion_c1_final PYTHON=/path/to/python bash train.sh [N_LVF] [N_LAV] [PARALLEL]
# Defaults reproduce the submitted ensemble: 17 LVF members (seeds 6001-6017) + 26 LAV members (seeds 7101-7126), 4 members at a time
# on the single GPU, then the 6 direct axis regressors and the POD-512 closure.  Wall-clock per stage is written to gpu_timing.json.
set -euo pipefail
cd "${FUSION_ROOT:?set FUSION_ROOT to the package root}"
PY=${PYTHON:-python}; NLVF=${1:-17}; NLAV=${2:-26}; PAR=${3:-4}
t() { date +%s; }; T0=$(t); declare -A W
log() { echo "=== $(date -u +%FT%TZ) $*"; }

log "members: LVF x$NLVF (B recipe + vessel + future inputs, Muon ratio 30)"; s=$(t)
$PY -u train_members.py LVF 6001 "$NLVF" --parallel "$PAR" --out members; W[members_LVF]=$(( $(t) - s ))
log "members: LAV x$NLAV (A recipe + vessel inputs, Muon ratio 200)"; s=$(t)
$PY -u train_members.py LAV 7101 "$NLAV" --parallel "$PAR" --out members; W[members_LAV]=$(( $(t) - s ))

log "direct axis regressors (6 x 256x3 MLPs, 60 epochs, train rows only)"; s=$(t)
TAGN=direct_vf60 SIZES=256x3 $PY -u results/analysis/scalar_probe_160/train_direct_vf.py 4,5,6,7,8,9 60; W[direct_regressors]=$(( $(t) - s ))

log "closure: POD-512 basis (+ autoencoder ceilings, as originally run) and the 64->448 closure network"; s=$(t)
$PY -u results/analysis/ae_ceiling_119/ae_ceiling.py results/analysis/ae_ceiling_119 8,16,32,64 120
$PY -u results/analysis/ae_ceiling_119/closure_ceiling.py 16,32,64 200
mkdir -p results/candidates/20261004-080/closure
cp results/analysis/ae_ceiling_119/basis512.npz results/analysis/ae_ceiling_119/closure64.pt results/candidates/20261004-080/closure/; W[closure]=$(( $(t) - s ))

W[total]=$(( $(t) - T0 ))
{ echo "{"; first=1; for k in "${!W[@]}"; do [ $first -eq 1 ] || echo ","; first=0; printf '  "%s_hours": %s' "$k" "$(awk -v s="${W[$k]}" 'BEGIN{printf "%.3f", s/3600}')"; done; echo; echo "}"; } > gpu_timing.json
cat gpu_timing.json; log "done"
