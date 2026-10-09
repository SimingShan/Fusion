#!/bin/bash
# GPU refinement-recipe sweep over several B-view cold checkpoints. Usage: refine_sweep_074.sh COLD_DIR [COLD_DIR ...]
set -uo pipefail
cd "${FUSION_ROOT:?set FUSION_ROOT to the package root}"
variants=(base amp1 amp05 liall liall_amp1)
for cold_dir in "$@"; do
  cold="$cold_dir/best.pt"; [ -f "$cold" ] || { echo "missing $cold"; continue; }
  tag=$(basename "$cold_dir")
  for v in "${variants[@]}"; do
    if [[ "$v" == liall* ]] && [ ! -f data/geometry_history_all_v5/metadata.json ]; then echo "skip $v: all-frame targets not ready"; continue; fi
    if [[ "$v" == liall* ]] && ! ${PYTHON:-python} -c "import json,sys; sys.exit(0 if json.load(open('data/geometry_history_all_v5/metadata.json')).get('complete') else 1)"; then echo "skip $v: all-frame targets incomplete"; continue; fi
    out="results/training/refine074/${tag}_${v}"
    [ -f "$out/native/val_score.json" ] && { echo "done $out"; continue; }
    bash scripts/optimization/refine_variant_073.sh "$cold" "$v" "$out" || echo "FAILED $tag $v"
  done
done
echo "=== sweep finished $(date -u +%FT%TZ) ==="
