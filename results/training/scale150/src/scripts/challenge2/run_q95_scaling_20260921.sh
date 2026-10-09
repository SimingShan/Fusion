#!/bin/bash
#SBATCH --job-name=c2_q95fix
#SBATCH --account=lu_lu
#SBATCH --partition=devel
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
set -euo pipefail
cd "${FUSION_ROOT:?set FUSION_ROOT to the package root}"
PY=${FUSION_ROOT}/${PYTHON:-python}
OUT=results/challenge2/c2_cnn_v3_main_q95fix
export OMP_NUM_THREADS=1
echo "start $(date -u +%FT%TZ) host $(hostname)"
$PY scripts/challenge2/fit_q95_scaling.py --n-shots 400 --frames-per-shot 8 --val-frames-per-shot 0 --workers 4 --out $OUT
$PY scripts/challenge2/apply_q95_scaling.py --fit $OUT/q95_scaling_fit.json --out $OUT --workers 4 --model loglin_fixed
echo "end $(date -u +%FT%TZ)"
