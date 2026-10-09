#!/bin/bash
cd "${FUSION_ROOT:?set FUSION_ROOT to the package root}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
${PYTHON:-python} scripts/challenge2/review_mast_scan_20260921.py 1
