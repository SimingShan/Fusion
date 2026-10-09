#!/bin/bash
# Conditioning/channel transfer-range audit (DIII-D train 200 x 8, dev 128 x 8, MAST 200 x 8), CPU only.
# Submit: sbatch --account=lu_lu --partition=bigmem --cpus-per-task=8 --mem=150G --time=02:00:00 \
#        -J c2_ranges -o logs/challenge2/c2_ranges_%j.log scripts/challenge2/run_c2_check_ranges_20260921.sh
set -euo pipefail
cd "${FUSION_ROOT:?set FUSION_ROOT to the package root}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
${PYTHON:-python} scripts/challenge2/check_c2_conditioning_ranges.py --n-train-shots 200 --n-mast-shots 200 --frames-per-shot 8 \
    --workers "$OMP_NUM_THREADS" --output results/challenge2/c2_cnn_conditioning_ranges.json
