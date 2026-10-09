#!/bin/bash
# Cold start on ALL 1,496,407 native training frames (B view), li refinement with native-mapped targets,
# then native dev + legacy predictions.  Usage: native_cold_079.sh SEED [EXTRA_COLD_ARGS...]
set -euo pipefail
cd "${FUSION_ROOT:?set FUSION_ROOT to the package root}"
seed="$1"; shift
workers="${SLURM_CPUS_PER_TASK:-2}"
export OMP_NUM_THREADS="$workers" OPENBLAS_NUM_THREADS="$workers" MKL_NUM_THREADS="$workers"
PY=${PYTHON:-python}
data=data/advanced_history_train_native_v7; pod=data/pod_history128_train_native_v7; li=data/targets_native_20260919
cold="results/training/ens079/N_s${seed}_cold"; ref="results/training/ens079/N_s${seed}"
echo "=== $(date -u +%FT%TZ) native cold start seed $seed ==="
$PY -u scripts/train_pod.py --data-dir $data --pod-dir $pod --output "$cold" --width 256 --depth 3 --epochs 300 \
   --batch-size 4096 --lr 0.001 --score-every 25 --ridge-alpha -1 --thomson-dropout 0.10 --seconds 7200 --workers "$workers" --seed "$seed" "$@"
echo "=== $(date -u +%FT%TZ) native li refinement ==="
$PY -u scripts/train_pod.py --data-dir $data --pod-dir $pod --output "$ref" --init "$cold/best.pt" --width 256 --depth 3 \
   --li-dir $li --li-sampled-only --li-weight 0.02 --geometry-weight 0.0 --frame-weighting amplitude --amplitude-power 1.0 --amplitude-max 8 \
   --thomson-dropout 0.10 --gradient-region envelope --epochs 100 --score-every 20 --batch-size 512 --lr 0.0001 --seconds 2400 --workers "$workers" --seed $((seed+1))
mkdir -p "$ref/native" "$ref/legacy"; ln -f "$ref/best.pt" "$ref/native/best.pt"; ln -f "$ref/best.pt" "$ref/legacy/best.pt"
$PY -u scripts/train_pod.py --predict-only --split val --data-dir data/advanced_history_val_native_v3 --output "$ref/native" --batch-size 512 --workers "$workers"
$PY -u scripts/train_pod.py --predict-only --split legacy_val --data-dir data/advanced_history_v2 --output "$ref/legacy" --batch-size 512 --workers "$workers"
echo "NATIVE_MEMBER $seed $(cat $ref/native/val_score.json | $PY -c 'import json,sys; d=json.load(sys.stdin); print(d["S"], d["consistency"])')"
