#!/bin/bash
# Fidelity of the synthetic free-boundary generator on DIII-D (labels exist): MAST generator physics on the DIII-D
# geometry in several arms, scored against EFIT with the official scorer functions.
# Submit: sbatch --account=lu_lu --partition=bigmem --cpus-per-task=32 --mem=200G --time=04:00:00 -J synfid_d3d \
#           -o logs/challenge2/synth_fidelity_d3d_%j.log scripts/analysis/synth_fidelity_d3d.sh
set -euo pipefail
cd "${FUSION_ROOT:?set FUSION_ROOT to the package root}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
OUT=${SF_OUT:-results/analysis/synth_fidelity_d3d}
mkdir -p "$OUT"
echo "== synth fidelity DIII-D: out=$OUT shots=${SF_SHOTS:-100} frames=${SF_FRAMES:-3} workers=${SLURM_CPUS_PER_TASK:-4}"
${PYTHON:-python} scripts/analysis/synth_fidelity_d3d.py --output "$OUT" --shots "${SF_SHOTS:-100}" --frames-per-shot "${SF_FRAMES:-3}" \
    --workers "${SLURM_CPUS_PER_TASK:-4}" --seed "${SF_SEED:-20260925}" --max-seconds "${SF_MAXSEC:-120}" ${SF_EXTRA:-}
echo "== done"
