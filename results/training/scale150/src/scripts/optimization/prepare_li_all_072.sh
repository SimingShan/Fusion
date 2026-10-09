#!/bin/bash
# Fixed-contour li / geometry supervision for ALL cached training frames (64 per shot).
set -euo pipefail
cd "${FUSION_ROOT:?set FUSION_ROOT to the package root}"
workers="${SLURM_CPUS_PER_TASK:-8}"
export OMP_NUM_THREADS="$workers" OPENBLAS_NUM_THREADS="$workers" MKL_NUM_THREADS="$workers" CUDA_VISIBLE_DEVICES=""
SRC=results/training/c1_sources_070
echo "=== $(date -u +%FT%TZ) li targets (A view, all frames) ==="
${PYTHON:-python} -u $SRC/scripts/optimization/prepare_li_targets.py --data-dir data/advanced_v1 --output data/li_full_all_v2 --frames-per-shot 64 --resume
echo "=== $(date -u +%FT%TZ) geometry targets (B view, all frames) ==="
${PYTHON:-python} -u $SRC/scripts/optimization/prepare_geometry_targets.py --data-dir data/advanced_history_v2 --source-li data/li_full_all_v2 --output data/geometry_history_all_v5
echo "=== $(date -u +%FT%TZ) done ==="
