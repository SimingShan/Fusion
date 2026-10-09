#!/bin/bash
# Train a chain of independently seeded ensemble members on one GPU.
# Usage: ensemble_chain_070.sh RECIPE SEED [SEED ...]   RECIPE in {A,B}
set -uo pipefail
cd "${FUSION_ROOT:?set FUSION_ROOT to the package root}"
recipe="$1"; shift
workers="${SLURM_CPUS_PER_TASK:-2}"
export OMP_NUM_THREADS="$workers" OPENBLAS_NUM_THREADS="$workers" MKL_NUM_THREADS="$workers"
SRC=results/training/c1_sources_070
PY=${PYTHON:-python}
out_root=results/training/ens070
if [ "$recipe" = "A" ]; then
  data=data/advanced_v1; pod=data/pod_scaled_v2; native=data/advanced_val_native_v3; li=data/li_full_v1
  cold_extra=(--batch-size 1024 --seconds 2400); ref_extra=(--thomson-dropout 0.0)
else
  data=data/advanced_history_v2; pod=data/pod_history128_robust_v4; native=data/advanced_history_val_native_v3; li=data/geometry_history_v4
  cold_extra=(--batch-size 2048 --thomson-dropout 0.10 --seconds 1800); ref_extra=(--thomson-dropout 0.10 --geometry-weight 0.0 --frame-weighting none)
fi
for seed in "$@"; do
  cold="$out_root/${recipe}_s${seed}_cold"; ref="$out_root/${recipe}_s${seed}"
  if [ -f "$ref/native/val_score.json" ]; then echo "member $recipe $seed already complete"; continue; fi
  echo "=== $(date -u +%FT%TZ) member $recipe seed $seed: cold start ==="
  $PY -u $SRC/scripts/train_pod.py --data-dir $data --pod-dir $pod --output "$cold" --width 256 --depth 3 \
     --epochs 600 --score-every 50 --ridge-alpha -1 --workers "$workers" --seed "$seed" "${cold_extra[@]}" || { echo "cold failed for seed $seed"; continue; }
  echo "=== $(date -u +%FT%TZ) member $recipe seed $seed: li refinement ==="
  $PY -u $SRC/scripts/train_pod.py --data-dir $data --pod-dir $pod --output "$ref" --init "$cold/best.pt" --width 256 --depth 3 \
     --li-dir $li --li-sampled-only --li-weight 0.02 --gradient-region envelope --epochs 100 --score-every 20 \
     --batch-size 512 --lr 0.0001 --seconds 1800 --workers "$workers" --seed $((seed+1)) "${ref_extra[@]}" || { echo "refine failed for seed $seed"; continue; }
  echo "=== $(date -u +%FT%TZ) member $recipe seed $seed: native + legacy predictions ==="
  mkdir -p "$ref/native" "$ref/legacy"; ln -f "$ref/best.pt" "$ref/native/best.pt"; ln -f "$ref/best.pt" "$ref/legacy/best.pt"
  $PY -u $SRC/scripts/train_pod.py --predict-only --split val --data-dir $native --output "$ref/native" --batch-size 512 --workers "$workers" || echo "native prediction failed for seed $seed"
  $PY -u $SRC/scripts/train_pod.py --predict-only --split legacy_val --data-dir $data --output "$ref/legacy" --batch-size 512 --workers "$workers" || echo "legacy prediction failed for seed $seed"
  $PY - "$ref" <<'PYEOF'
import json,sys
from pathlib import Path
r=Path(sys.argv[1]); out={}
for k,f in (("capped",r/"val_score.json"),("native",r/"native/val_score.json"),("legacy",r/"legacy/legacy_val_score.json")):
    if f.exists():
        d=json.loads(f.read_text()); out[k]={x:d[x] for x in ("S","r2_psi","r2_qb","dlcfs","consistency")}; out[k]["r2_cons_each"]=d.get("r2_cons_each")
print("MEMBER_SUMMARY", r.name, json.dumps(out))
PYEOF
done
echo "=== chain finished $(date -u +%FT%TZ) ==="
