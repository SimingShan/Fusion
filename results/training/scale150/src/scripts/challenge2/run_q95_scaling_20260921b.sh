#!/bin/bash
#SBATCH --job-name=c2_q95fix_b
#SBATCH --account=lu_lu
#SBATCH --partition=devel
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
# Re-fit with the eps-PINNED model family (q95_scaling_common.py) and re-apply to MAST with pinned_p075.
# The free-eps model (loglin_fixed) of run_q95_scaling_20260921.sh collapsed to q95 ~ 1.5 on MAST (87% at the 2.0 floor).
set -euo pipefail
cd "${FUSION_ROOT:?set FUSION_ROOT to the package root}"
PY=${FUSION_ROOT}/${PYTHON:-python}
OUT=results/challenge2/c2_cnn_v3_main_q95fix
MODEL=${MODEL:-pinned_p075}
export OMP_NUM_THREADS=1
echo "start $(date -u +%FT%TZ) host $(hostname) model $MODEL"
$PY scripts/challenge2/fit_q95_scaling.py --n-shots 400 --frames-per-shot 8 --val-frames-per-shot 0 --workers 4 --out $OUT
$PY scripts/challenge2/apply_q95_scaling.py --fit $OUT/q95_scaling_fit.json --out $OUT --workers 4 --model $MODEL
echo "end $(date -u +%FT%TZ)"
