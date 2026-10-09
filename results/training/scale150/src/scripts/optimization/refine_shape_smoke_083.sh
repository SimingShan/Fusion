#!/bin/bash
# Shape-loss smoke/ablation 083: refine the B-view cold checkpoint results/training/ens070/B_s3107_cold/best.pt for
# 12 CPU epochs with the liall_amp1 recipe (all-frame li supervision, amplitude frame weights) plus the new geometry
# flags, then predict and score the native dev fold (data/advanced_history_val_native_v3).
# Arms: base (new flags off) | shape (--geometry-weight 0.02 --boundary-loss-weight 1 --boundary-extremal-weight 4)
#       | shape_margin (shape + --axis-margin-weight 0.02).  12 epochs is a direction check, not a final number.
# Frozen-style: `submit` copies scripts/ and fusion/ (plus a starter/ symlink) into results/training/shape083/src and
# every job runs the frozen copies only, so later edits to the live tree cannot affect running or queued arms.
# Usage:
#   scripts/optimization/refine_shape_smoke_083.sh submit [ARM ...]   # freeze sources, sbatch the arms (default: all three)
#   results/training/shape083/src/refine_shape_smoke_083.sh run ARM   # inside a job
set -euo pipefail
cd "${FUSION_ROOT:?set FUSION_ROOT to the package root}"
ROOT_OUT=results/training/shape083
SRC=$ROOT_OUT/src
mode=${1:-submit}
if [ "$mode" = submit ]; then
  shift; arms=("$@"); [ ${#arms[@]} -eq 0 ] && arms=(base shape shape_margin)
  mkdir -p "$SRC" logs/refine
  rm -rf "$SRC/scripts" "$SRC/fusion"; cp -r scripts fusion "$SRC/"; find "$SRC" -name __pycache__ -type d -prune -exec rm -rf {} +
  ln -sfn "$PWD/starter" "$SRC/starter"
  cp scripts/optimization/refine_shape_smoke_083.sh "$SRC/refine_shape_smoke_083.sh"
  sha256sum scripts/train_pod.py fusion/geometry_loss.py fusion/li_loss.py scripts/optimization/loss_terms_083.py > "$SRC/SOURCES.sha256"
  for arm in "${arms[@]}"; do
    # devel (cpu=10, mem=70G per user) is the default; SMOKE_PARTITION=bigmem SMOKE_MEM=150G (bigmem minimum) when devel is saturated.
    sbatch --account=lu_lu --partition="${SMOKE_PARTITION:-devel}" --cpus-per-task=3 --mem="${SMOKE_MEM:-32G}" --time=03:00:00 -J "shape083_$arm" \
      -o "logs/refine/shape083_${arm}_%j.log" "$SRC/refine_shape_smoke_083.sh" run "$arm"
  done
  exit 0
fi
[ "$mode" = run ] || { echo "usage: $0 submit [ARM ...] | run ARM"; exit 2; }
arm=$2; out=$ROOT_OUT/$arm
workers=${SLURM_CPUS_PER_TASK:-3}
export OMP_NUM_THREADS=$workers OPENBLAS_NUM_THREADS=$workers MKL_NUM_THREADS=$workers CUDA_VISIBLE_DEVICES=""
PY=${PYTHON:-python}; TP=$SRC/scripts/train_pod.py
cold=results/training/ens070/B_s3107_cold/best.pt
data=data/advanced_history_v2; pod=data/pod_history128_robust_v4; native=data/advanced_history_val_native_v3; li=data/geometry_history_all_v5
case "$arm" in
  base)         extra=(--geometry-weight 0.0) ;;
  shape)        extra=(--geometry-weight 0.02 --boundary-loss-weight 1 --boundary-extremal-weight 4) ;;
  shape_margin) extra=(--geometry-weight 0.02 --boundary-loss-weight 1 --boundary-extremal-weight 4 --axis-margin-weight 0.02) ;;
  *) echo "unknown arm $arm"; exit 2 ;;
esac
mkdir -p "$out"
echo "=== $(date -u +%FT%TZ) shape083 arm=$arm workers=$workers host=$(hostname) job=${SLURM_JOB_ID:-none} ==="; cat "$SRC/SOURCES.sha256"
# Per-term loss magnitudes at the start (init checkpoint), unweighted and amplitude-weighted.
$PY -u "$SRC/scripts/optimization/loss_terms_083.py" --checkpoint "$cold" --pod-dir $pod --li-dir $li --batches 8 --batch-size 512 \
  --boundary-extremal-weight 4 --axis-margin-ratio 0.5 --workers "$workers" --output "$out/loss_terms_start.json"
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
h = json.loads((r / "history.json").read_text())
out["capped_S_by_epoch"] = {e["epoch"]: e["S"] for e in h if "S" in e}
steps = [b["elapsed_seconds"] - a["elapsed_seconds"] for a, b in zip(h, h[1:])]
out["seconds_per_epoch_mean"] = sum(steps) / max(len(steps), 1); out["train_loss_by_epoch"] = [round(e["train_loss"], 6) for e in h]
for k in ("best", "e12"):
    f = r / f"native_{k}" / "val_score.json"
    if f.exists():
        d = json.loads(f.read_text()); out[f"native_{k}"] = {x: d[x] for x in ("S", "r2_psi", "r2_qb", "dlcfs", "consistency")}
        out[f"native_{k}"]["cons"] = d.get("r2_cons_each")
print("SHAPE083_SUMMARY", json.dumps(out))
(r / "summary.json").write_text(json.dumps(out, indent=2) + "\n")
PYEOF
echo "=== $(date -u +%FT%TZ) done arm=$arm ==="
