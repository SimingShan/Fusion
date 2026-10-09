#!/bin/bash
# v3 synthetic free-boundary MAST equilibria: real public inputs + UNOBSERVED perturbations (passive-structure
# currents in the solve vacuum field, 3% coil-current noise for the solve only, sampled Thomson p/pe ratio) and a
# double-null sampling mode. Record inputs stay the real coil currents (+ P5/P6 control corrections).
# Submit: sbatch --account=lu_lu --partition=bigmem --cpus-per-task=32 --mem=200G --time=04:00:00 -J synth_v3 \
#           -o logs/challenge2/synth_v3_%j.log scripts/challenge2/generate_synthetic_mast_v3_20260923.sh
set -euo pipefail
cd "${FUSION_ROOT:?set FUSION_ROOT to the package root}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
OUT=${SYN_OUT:-data/c2_synth_mast_v3}
TARGET=${SYN_TARGET:-40000}
WORKERS=${SLURM_CPUS_PER_TASK:-4}
RANGES='{"beta0": [0.15, 0.75], "alpha_p": [1.0, 2.5], "gamma_p": [1.0, 3.0], "alpha_f": [1.2, 3.0], "gamma_f": [0.4, 1.4]}'
mkdir -p "$OUT"
echo "== generate v3: out=$OUT target=$TARGET workers=$WORKERS"
${PYTHON:-python} scripts/challenge2/generate_synthetic_mast.py --output "$OUT" --target "$TARGET" --workers "$WORKERS" \
    --seed "${SYN_SEED:-2029}" --ranges "$RANGES" --r-target-range 0.76 0.90 --z-std 0.02 --z-max 0.05 \
    --max-attempts 4 --oversample 1.3 --time-limit "${SYN_TIME_LIMIT:-12000}" \
    --vessel-current-kA "${SYN_VESSEL_KA:-30}" --vessel-asym-frac "${SYN_VESSEL_ASYM:-0.35}" --vessel-share-std 0.4 \
    --coil-noise 0.03 --pe-ratio-range 1.5 2.5 --double-null-fraction "${SYN_DN_FRAC:-0.5}" --dn-asym-std 0.01 --dn-asym-max 0.03 \
    --v3-metadata --compare-report data/c2_synth_mast_v1/report.json ${SYN_EXTRA:-}
echo "== check"
${PYTHON:-python} scripts/challenge2/check_synthetic_mast_v3.py --data-dir "$OUT" --output results/challenge2/c2_synth_mast_v3 --samples 240 --reference-samples 800
cp "$OUT/report.json" "$OUT/manifest.json" results/challenge2/c2_synth_mast_v3/
echo "== done"
