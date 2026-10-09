#!/bin/bash
#SBATCH --job-name=c1_anat083
cd "${FUSION_ROOT:?set FUSION_ROOT to the package root}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
echo "start $(date)"
${PYTHON:-python} scripts/analysis/c1_error_anatomy_083.py all --workers 16
echo "end $(date)"
