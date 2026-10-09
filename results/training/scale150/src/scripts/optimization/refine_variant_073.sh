#!/bin/bash
# Refine a B-view cold checkpoint with a named recipe variant, then predict the native dev fold.
# Usage: refine_variant_073.sh COLD_BEST_PT VARIANT OUT_DIR   VARIANT in {base, amp05, amp1, amp15, amp2, liall, liall_amp1}
set -euo pipefail
cd "${FUSION_ROOT:?set FUSION_ROOT to the package root}"
cold="$1"; variant="$2"; out="$3"
workers="${SLURM_CPUS_PER_TASK:-3}"
export OMP_NUM_THREADS="$workers" OPENBLAS_NUM_THREADS="$workers" MKL_NUM_THREADS="$workers"
PY=${PYTHON:-python}
data=data/advanced_history_v2; pod=data/pod_history128_robust_v4; native=data/advanced_history_val_native_v3
common=(--data-dir $data --pod-dir $pod --output "$out" --init "$cold" --li-sampled-only --li-weight 0.02 --geometry-weight 0.0 --thomson-dropout 0.10 --gradient-region envelope --score-every 20 --batch-size 512 --lr 0.0001 --workers "$workers" --seed 2029)
case "$variant" in
  base)       extra=(--li-dir data/geometry_history_v4 --epochs 100 --seconds 1800 --frame-weighting none) ;;
  amp1)       extra=(--li-dir data/geometry_history_v4 --epochs 100 --seconds 1800 --frame-weighting amplitude --amplitude-power 1.0 --amplitude-max 8) ;;
  amp05)      extra=(--li-dir data/geometry_history_v4 --epochs 100 --seconds 1800 --frame-weighting amplitude --amplitude-power 0.5 --amplitude-max 4) ;;
  amp15)      extra=(--li-dir data/geometry_history_v4 --epochs 100 --seconds 1800 --frame-weighting amplitude --amplitude-power 1.5 --amplitude-max 12) ;;
  amp2)       extra=(--li-dir data/geometry_history_v4 --epochs 100 --seconds 1800 --frame-weighting amplitude --amplitude-power 2.0 --amplitude-max 16) ;;
  liall)      extra=(--li-dir data/geometry_history_all_v5 --epochs 30 --seconds 2400 --frame-weighting none) ;;
  liall_amp1_e60) extra=(--li-dir data/geometry_history_all_v5 --epochs 60 --seconds 4200 --frame-weighting amplitude --amplitude-power 1.0 --amplitude-max 8) ;;
  liall_e60)  extra=(--li-dir data/geometry_history_all_v5 --epochs 60 --seconds 4200 --frame-weighting none) ;;
  liall_amp1) extra=(--li-dir data/geometry_history_all_v5 --epochs 30 --seconds 2400 --frame-weighting amplitude --amplitude-power 1.0 --amplitude-max 8) ;;
  *) echo "unknown variant $variant"; exit 2 ;;
esac
# Width/depth from the checkpoint override the defaults above.
read -r W D < <($PY -c "import torch,sys; s=torch.load('$cold',map_location='cpu',weights_only=False); k=s['model_kwargs']; print(k['width'], k['depth'])")
echo "=== $(date -u +%FT%TZ) refine $variant from $cold (width $W depth $D) -> $out ==="
$PY -u scripts/train_pod.py "${common[@]}" --width "$W" --depth "$D" "${extra[@]}"
mkdir -p "$out/native"; ln -f "$out/best.pt" "$out/native/best.pt"
$PY -u scripts/train_pod.py --predict-only --split val --data-dir $native --output "$out/native" --batch-size 512 --workers "$workers"
$PY - "$out" "$variant" <<'PYEOF'
import json,sys
from pathlib import Path
r=Path(sys.argv[1]); out={}
for k,f in (("capped",r/"val_score.json"),("native",r/"native/val_score.json")):
    if f.exists():
        d=json.loads(f.read_text()); out[k]={x:d[x] for x in ("S","r2_psi","r2_qb","dlcfs","consistency")}; out[k]["cons"]=d.get("r2_cons_each")
print("VARIANT_SUMMARY", sys.argv[2], r.name, json.dumps(out))
PYEOF
