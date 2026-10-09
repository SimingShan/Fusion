#!/bin/bash
# Band-loss smoke 086: refine the B-view cold checkpoint results/training/ens070/B_s3107_cold/best.pt for 12 CPU epochs
# with the liall_amp1 recipe (all-frame li supervision, amplitude frame weights) plus the new pixel-weighted
# reconstructed-field loss (--band-loss-weight, scripts/train_pod.py / fusion/geometry_loss.py band_pixel_weights),
# then predict and score the native dev fold (data/advanced_history_val_native_v3).
# Arms: ref   (new flags off)
#       bandA (--band-loss-weight 1 --band-weight 8 --core-weight 8)
#       bandB (--band-loss-weight 3 --band-weight 8 --core-weight 0)
#       bandC (--band-loss-weight 20 --band-weight 8 --core-weight 8; extra arm, band term ~O(coefficient loss))
# 12 epochs single seed is a direction check: differences below ~1e-4 S are noise.
# Frozen-style: `submit` copies scripts/ and fusion/ (plus a starter/ symlink) into results/training/band086/src and
# every job runs the frozen copies only, so later edits to the live tree cannot affect running or queued arms.
# Usage:
#   scripts/optimization/refine_band_smoke_086.sh submit [ARM ...]   # freeze sources, sbatch the arms (default: ref bandA bandB bandC)
#   results/training/band086/src/refine_band_smoke_086.sh run ARM   # inside a job
set -euo pipefail
cd "${FUSION_ROOT:?set FUSION_ROOT to the package root}"
ROOT_OUT=results/training/band086
SRC=$ROOT_OUT/src
mode=${1:-submit}
if [ "$mode" = submit ]; then
  shift; arms=("$@"); [ ${#arms[@]} -eq 0 ] && arms=(ref bandA bandB bandC)
  mkdir -p "$SRC" logs/refine
  rm -rf "$SRC/scripts" "$SRC/fusion"; cp -r scripts fusion "$SRC/"; find "$SRC" -name __pycache__ -type d -prune -exec rm -rf {} +
  ln -sfn "$PWD/starter" "$SRC/starter"
  cp scripts/optimization/refine_band_smoke_086.sh "$SRC/refine_band_smoke_086.sh"
  sha256sum scripts/train_pod.py fusion/geometry_loss.py fusion/li_loss.py fusion/amplitude_weights.py > "$SRC/SOURCES.sha256"
  for arm in "${arms[@]}"; do
    sbatch --account=lu_lu --partition="${SMOKE_PARTITION:-bigmem}" --cpus-per-task=8 --mem="${SMOKE_MEM:-150G}" --time=03:00:00 -J "band086_$arm" \
      -o "logs/refine/band086_${arm}_%j.log" "$SRC/refine_band_smoke_086.sh" run "$arm"
  done
  exit 0
fi
[ "$mode" = run ] || { echo "usage: $0 submit [ARM ...] | run ARM"; exit 2; }
arm=$2; out=$ROOT_OUT/$arm
workers=${SLURM_CPUS_PER_TASK:-8}
export OMP_NUM_THREADS=$workers OPENBLAS_NUM_THREADS=$workers MKL_NUM_THREADS=$workers CUDA_VISIBLE_DEVICES=""
PY=${PYTHON:-python}; TP=$SRC/scripts/train_pod.py
cold=results/training/ens070/B_s3107_cold/best.pt
data=data/advanced_history_v2; pod=data/pod_history128_robust_v4; native=data/advanced_history_val_native_v3; li=data/geometry_history_all_v5
case "$arm" in
  ref)   extra=() ;;
  bandA) extra=(--band-loss-weight 1 --band-weight 8 --core-weight 8) ;;
  bandB) extra=(--band-loss-weight 3 --band-weight 8 --core-weight 0) ;;
  bandC) extra=(--band-loss-weight 20 --band-weight 8 --core-weight 8) ;;
  *) echo "unknown arm $arm"; exit 2 ;;
esac
mkdir -p "$out"
echo "=== $(date -u +%FT%TZ) band086 arm=$arm workers=$workers host=$(hostname) job=${SLURM_JOB_ID:-none} extra=${extra[*]:-none} ==="; cat "$SRC/SOURCES.sha256"
$PY -u "$TP" --data-dir $data --pod-dir $pod --output "$out" --init "$cold" --width 256 --depth 3 \
  --li-dir $li --li-sampled-only --li-weight 0.02 --frame-weighting amplitude --amplitude-power 1.0 --amplitude-max 8 \
  --gradient-region envelope --thomson-dropout 0.10 --epochs 12 --seconds 7800 --score-every 6 --batch-size 512 --lr 0.0001 \
  --workers "$workers" --seed 2029 "${extra[@]}"
# Native dev-fold prediction of the best-scored checkpoint and of the epoch-12 snapshot (arms compare at equal epochs).
for tag in best e12; do
  d="$out/native_$tag"; mkdir -p "$d"
  if [ "$tag" = best ]; then src_pt="$out/best.pt"; else src_pt="$out/checkpoints/epoch_0012.pt"; fi
  [ -f "$src_pt" ] || { echo "missing $src_pt"; continue; }
  ln -f "$src_pt" "$d/best.pt"
  $PY -u "$TP" --predict-only --split val --data-dir $native --output "$d" --batch-size 512 --workers "$workers"
done
$PY - "$out" "$arm" <<'PYEOF'
import json, sys
from pathlib import Path
r = Path(sys.argv[1]); out = {"arm": sys.argv[2]}
h = json.loads((r / "history.json").read_text()); run = json.loads((r / "run.json").read_text())
out["capped_S_by_epoch"] = {e["epoch"]: e["S"] for e in h if "S" in e}
steps = [b["elapsed_seconds"] - a["elapsed_seconds"] for a, b in zip(h, h[1:])]
out["seconds_per_epoch_mean"] = sum(steps) / max(len(steps), 1); out["train_loss_by_epoch"] = [round(e["train_loss"], 6) for e in h]
out["train_coefficient_loss_by_epoch"] = [e.get("train_coefficient_loss") for e in h]
out["train_band_loss_by_epoch"] = [e.get("train_band_loss") for e in h]
out["band_loss_start"] = run.get("band_loss")
for k in ("best", "e12"):
    f = r / f"native_{k}" / "val_score.json"
    if f.exists():
        d = json.loads(f.read_text()); out[f"native_{k}"] = {x: d[x] for x in ("S", "r2_psi", "r2_qb", "dlcfs", "consistency")}
        out[f"native_{k}"]["cons"] = d.get("r2_cons_each")
print("BAND086_SUMMARY", json.dumps(out))
(r / "summary.json").write_text(json.dumps(out, indent=2) + "\n")
PYEOF
echo "=== $(date -u +%FT%TZ) done arm=$arm ==="
