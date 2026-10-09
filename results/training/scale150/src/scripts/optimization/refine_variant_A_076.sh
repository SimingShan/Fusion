#!/bin/bash
# Refine an A-view (scaled features) cold checkpoint with a named recipe variant, then predict the native dev fold.
# Usage: refine_variant_A_076.sh COLD_BEST_PT VARIANT OUT_DIR   VARIANT in {base, amp05, amp1, liall, liall_amp1}
set -euo pipefail
cd "${FUSION_ROOT:?set FUSION_ROOT to the package root}"
cold="$1"; variant="$2"; out="$3"
workers="${SLURM_CPUS_PER_TASK:-4}"
export OMP_NUM_THREADS="$workers" OPENBLAS_NUM_THREADS="$workers" MKL_NUM_THREADS="$workers"
PY=${PYTHON:-python}
data=data/advanced_v1; pod=data/pod_scaled_v2; native=data/advanced_val_native_v3
common=(--data-dir $data --pod-dir $pod --output "$out" --init "$cold" --li-sampled-only --li-weight 0.02 --gradient-region envelope --score-every 20 --batch-size 512 --lr 0.0001 --workers "$workers" --seed 2026)
case "$variant" in
  base)       extra=(--li-dir data/li_full_v1 --epochs 100 --seconds 1800 --frame-weighting none) ;;
  amp05)      extra=(--li-dir data/li_full_v1 --epochs 100 --seconds 1800 --frame-weighting amplitude --amplitude-power 0.5 --amplitude-max 4) ;;
  amp1)       extra=(--li-dir data/li_full_v1 --epochs 100 --seconds 1800 --frame-weighting amplitude --amplitude-power 1.0 --amplitude-max 8) ;;
  liall)      extra=(--li-dir data/li_full_all_v2 --epochs 30 --seconds 2400 --frame-weighting none) ;;
  liall_amp1_e60) extra=(--li-dir data/li_full_all_v2 --epochs 60 --seconds 4200 --frame-weighting amplitude --amplitude-power 1.0 --amplitude-max 8) ;;
  liall_e60)  extra=(--li-dir data/li_full_all_v2 --epochs 60 --seconds 4200 --frame-weighting none) ;;
  liall_amp1) extra=(--li-dir data/li_full_all_v2 --epochs 30 --seconds 2400 --frame-weighting amplitude --amplitude-power 1.0 --amplitude-max 8) ;;
  *) echo "unknown variant $variant"; exit 2 ;;
esac
read -r W D < <($PY -c "import torch; s=torch.load('$cold',map_location='cpu',weights_only=False); k=s['model_kwargs']; print(k['width'], k['depth'])")
echo "=== $(date -u +%FT%TZ) refine A $variant from $cold (width $W depth $D) -> $out ==="
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
