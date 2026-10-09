#!/bin/bash
# Re-refine a list of cold checkpoints with one recipe variant (B view unless the dir name starts with A_), then predict
# the native dev fold and the legacy fold for blending.  Usage: rerefine_078.sh VARIANT COLD_DIR [COLD_DIR ...]
set -uo pipefail
cd "${FUSION_ROOT:?set FUSION_ROOT to the package root}"
variant="$1"; shift
workers="${SLURM_CPUS_PER_TASK:-2}"
PY=${PYTHON:-python}
for cold_dir in "$@"; do
  cold="$cold_dir/best.pt"; [ -f "$cold" ] || { echo "missing $cold"; continue; }
  tag=$(basename "$cold_dir" | sed 's/_cold$//')
  out="results/training/ens078_${variant}/${tag}"
  if [ -f "$out/legacy/legacy_val_score.json" ]; then echo "done $out"; continue; fi
  if [[ "$tag" == A_* ]] || [[ "$cold_dir" == *pod_scaled_v2* ]]; then
    bash scripts/optimization/refine_variant_A_076.sh "$cold" "$variant" "$out" || { echo "FAILED $tag"; continue; }
    data=data/advanced_v1
  else
    bash scripts/optimization/refine_variant_073.sh "$cold" "$variant" "$out" || { echo "FAILED $tag"; continue; }
    data=data/advanced_history_v2
  fi
  mkdir -p "$out/legacy"; ln -f "$out/best.pt" "$out/legacy/best.pt"
  export OMP_NUM_THREADS="$workers" OPENBLAS_NUM_THREADS="$workers" MKL_NUM_THREADS="$workers"
  $PY -u scripts/train_pod.py --predict-only --split legacy_val --data-dir $data --output "$out/legacy" --batch-size 512 --workers "$workers" || echo "legacy prediction failed for $tag"
  echo "REREFINED $variant $tag $(cat $out/native/val_score.json 2>/dev/null | $PY -c 'import json,sys; d=json.load(sys.stdin); print(d["S"], d["consistency"])')"
done
echo "=== rerefine $variant finished $(date -u +%FT%TZ) ==="
