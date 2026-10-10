#!/bin/bash
# GPU training entrypoint (from scratch; the only GPU-budgeted part).  Requires prepare.sh to have run.
#   FUSION_ROOT=/path/to/fusion_c1_final PYTHON=/path/to/python bash train.sh
# Defaults reproduce the submitted ensemble: 24 LVF members (seeds 6001-6017, 6101-6107) + 40 LAV members (seeds 7101-7126, 7201-7214),
# 4 members at a time on the single GPU, then the 6 direct regressors and the POD-512 closure.  Wall-clock per stage -> gpu_timing.json.
# Overrides: LVF_SEEDS / LAV_SEEDS (start:count[,start:count...]), PARALLEL (members per GPU).
set -euo pipefail
cd "${FUSION_ROOT:?set FUSION_ROOT to the package root}"
PY=${PYTHON:-python}; PAR=${PARALLEL:-4}
LVF_SEEDS=${LVF_SEEDS:-6001:17,6101:7}; LAV_SEEDS=${LAV_SEEDS:-7101:26,7201:14}   # member seed ranges (start:count)
t() { date +%s; }; T0=$(t); declare -A W
log() { echo "=== $(date -u +%FT%TZ) $*"; }

log "members: LVF seeds $LVF_SEEDS (B recipe + vessel + future inputs, Muon ratio 30)"; s=$(t)
$PY -u train_members.py LVF 0 0 --seeds "$LVF_SEEDS" --parallel "$PAR" --out members; W[members_LVF]=$(( $(t) - s ))
log "members: LAV seeds $LAV_SEEDS (A recipe + vessel inputs, Muon ratio 200)"; s=$(t)
$PY -u train_members.py LAV 0 0 --seeds "$LAV_SEEDS" --parallel "$PAR" --out members; W[members_LAV]=$(( $(t) - s ))

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
