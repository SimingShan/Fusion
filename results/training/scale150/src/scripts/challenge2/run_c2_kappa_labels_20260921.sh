#!/bin/bash
# Shot-level label elongation for the kappa extrapolation stress test (devel partition, ~10 min).
# Usage: sbatch --account=lu_lu --partition=devel --cpus-per-task=4 --mem=16G --time=01:00:00 \
#        -o logs/challenge2/c2_kappa_%j.log scripts/challenge2/run_c2_kappa_labels_20260921.sh
set -euo pipefail
cd "${FUSION_ROOT:?set FUSION_ROOT to the package root}"
export OMP_NUM_THREADS=1
${PYTHON:-python} scripts/challenge2/compute_kappa_labels.py --train-frames 4 --val-frames 8 --low-fraction 0.6 \
    --out results/challenge2/c2_cnn_kappa --split-dir data/c2_cnn_kappa_split --workers "${SLURM_CPUS_PER_TASK:-4}"
