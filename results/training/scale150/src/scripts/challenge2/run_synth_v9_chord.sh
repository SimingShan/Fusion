#!/bin/bash
# v9 synthetic MAST generation (iteration 3 design): v3 realism options (vessel 30 kA, coil noise 3 %, pe ratio 1.5-2.5, double-null 0.5)
# PLUS the v9 options: position from the frame's own MIDPLANE THOMSON CHORD (label-free, --position-mode chord: the core-centre
# estimator of the equilibrium's own chord pressure is iterated to the measured value, beta0 balanced so that the REAL vacuum field
# holds the plasma, |dI5| < 1.5 kA-turn), edge/pedestal current, reject-unbalanced (beta0 at a bound), beta_p consistency with the
# Thomson proxy (factor 2.5).
# Env: OUT, TARGET, SEED, MAX_SHOTS (0 = all), TIME_LIMIT (s), VAL_SHOTS, CHORD_MATCH (axis|rgeo), EXTRA. Launch FROZEN copies only.
# Submit (bigmem): sbatch --account=lu_lu --partition=bigmem --cpus-per-task=32 --mem=200G --time=04:00:00 -J synth_v9 \
#                    -o logs/challenge2/synth_v9_%j.log --export=ALL,OUT=data/c2_synth_mast_v9,TARGET=40000,SEED=2030 <frozen copy>
set -euo pipefail
cd "${FUSION_ROOT:?set FUSION_ROOT to the package root}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
OUT=${OUT:-data/c2_synth_mast_v9}
mkdir -p "$OUT"
echo "== synth v9 chord: out=$OUT target=${TARGET:-40000} seed=${SEED:-2030} workers=${SLURM_CPUS_PER_TASK:-4} max_shots=${MAX_SHOTS:-0} match=${CHORD_MATCH:-axis} extra='${EXTRA:-}'"
${PYTHON:-python} scripts/challenge2/generate_synthetic_mast.py --output "$OUT" --target "${TARGET:-40000}" --workers "${SLURM_CPUS_PER_TASK:-4}" \
    --seed "${SEED:-2030}" --val-shots "${VAL_SHOTS:-60}" --max-shots "${MAX_SHOTS:-0}" --max-attempts 4 --oversample "${OVERSAMPLE:-1.3}" \
    --ranges '{"alpha_f": [1.2, 3.0], "gamma_f": [0.4, 1.4]}' --r-target-range 0.76 0.90 --z-std 0.02 --z-max 0.05 \
    --vessel-current-kA 30 --vessel-asym-frac 0.35 --vessel-share-std 0.4 --coil-noise 0.03 --pe-ratio-range 1.5 2.5 \
    --double-null-fraction 0.5 --dn-asym-std 0.01 --dn-asym-max 0.03 --v3-metadata \
    --position-mode chord --chord-match "${CHORD_MATCH:-axis}" --chord-steps 2 --chord-tol 0.005 --chord-accept-tol 0.02 --r-init-range 0.60 1.05 \
    --proxy-ratio-range 1.5 2.5 --edge-current --edge-amp-range 0 0.3 --edge-w-range 0.03 0.12 \
    --reject-unbalanced --beta-p-consistency 2.5 --time-limit "${TIME_LIMIT:-12600}" --compare-report data/c2_synth_mast_v3/report.json ${EXTRA:-}
echo "== done"
