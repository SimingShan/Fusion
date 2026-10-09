#!/bin/bash
#SBATCH --job-name=c1_chk083
cd "${FUSION_ROOT:?set FUSION_ROOT to the package root}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
echo "start $(date)"
${PYTHON:-python} scripts/analysis/c1_anat083_check_fluxw.py
echo "end $(date)"
