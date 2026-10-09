#!/bin/bash
# v9 synthetic MAST generation: v3 realism options (vessel 30 kA, coil noise 3 %, pe ratio 1.5-2.5, double-null 0.5) PLUS the v9
# options (position from force balance with the label-free Thomson beta_p proxy, edge/pedestal current, reject-unbalanced).
# Parametrised by env vars so that pilots and the final run use frozen copies of this file:
#   OUT (output dir), TARGET, SEED, MAX_SHOTS (0 = all), TIME_LIMIT (s), VAL_SHOTS, EXTRA (extra CLI flags)
# Submit (bigmem): sbatch --account=lu_lu --partition=bigmem --cpus-per-task=32 --mem=200G --time=04:00:00 -J synth_v9 \
#                    -o logs/challenge2/synth_v9_%j.log --export=ALL,OUT=data/c2_synth_mast_v9,TARGET=40000,SEED=2030 <frozen copy>
set -euo pipefail
cd "${FUSION_ROOT:?set FUSION_ROOT to the package root}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
OUT=${OUT:-data/c2_synth_mast_v9}
mkdir -p "$OUT"
echo "== synth v9: out=$OUT target=${TARGET:-40000} seed=${SEED:-2030} workers=${SLURM_CPUS_PER_TASK:-4} max_shots=${MAX_SHOTS:-0} extra='${EXTRA:-}'"
${PYTHON:-python} scripts/challenge2/generate_synthetic_mast.py --output "$OUT" --target "${TARGET:-40000}" --workers "${SLURM_CPUS_PER_TASK:-4}" \
    --seed "${SEED:-2030}" --val-shots "${VAL_SHOTS:-60}" --max-shots "${MAX_SHOTS:-0}" --max-attempts 4 --oversample "${OVERSAMPLE:-1.3}" \
    --ranges '{"alpha_f": [1.2, 3.0], "gamma_f": [0.4, 1.4]}' --r-target-range 0.76 0.90 --z-std 0.02 --z-max 0.05 \
    --vessel-current-kA 30 --vessel-asym-frac 0.35 --vessel-share-std 0.4 --coil-noise 0.03 --pe-ratio-range 1.5 2.5 \
    --double-null-fraction 0.5 --dn-asym-std 0.01 --dn-asym-max 0.03 --v3-metadata \
    --position-mode balance --proxy-ratio-range 1.5 2.5 --beta-p-jitter "${BP_JITTER:-0.3}" --edge-current --edge-amp-range 0 0.3 --edge-w-range 0.03 0.12 \
    --reject-unbalanced --time-limit "${TIME_LIMIT:-12000}" --compare-report data/c2_synth_mast_v3/report.json ${EXTRA:-}
echo "== done"
