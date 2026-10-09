#!/bin/bash
# Synthetic free-boundary MAST equilibria on real public inputs (fusion/c2_synthetic.py).
# Pilot:  SYN_OUT=results/challenge2/c2_synth_mast_pilot SYN_TARGET=1500 sbatch --account=lu_lu --partition=bigmem \
#           --cpus-per-task=32 --mem=200G --time=01:00:00 -J synth_pilot -o logs/challenge2/synth_pilot_%j.log \
#           scripts/challenge2/generate_synthetic_mast_20260922.sh
# Full:   SYN_OUT=data/c2_synth_mast_v1 SYN_TARGET=20000 sbatch ... --time=06:00:00 -J synth_full -o logs/challenge2/synth_full_%j.log ...
set -euo pipefail
cd "${FUSION_ROOT:?set FUSION_ROOT to the package root}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
OUT=${SYN_OUT:-data/c2_synth_mast_v1}
TARGET=${SYN_TARGET:-20000}
WORKERS=${SLURM_CPUS_PER_TASK:-4}
RANGES=${SYN_RANGES:-'{"beta0": [0.15, 0.75], "alpha_p": [1.0, 2.5], "gamma_p": [1.0, 3.0], "alpha_f": [1.2, 3.0], "gamma_f": [0.4, 1.4]}'}
RT=${SYN_R_TARGET:-"0.76 0.90"}
ZSTD=${SYN_Z_STD:-0.02}
ZMAX=${SYN_Z_MAX:-0.05}
EXTRA=${SYN_EXTRA:-}
mkdir -p "$OUT"
echo "== generate: out=$OUT target=$TARGET workers=$WORKERS ranges=$RANGES r_target=$RT z_std=$ZSTD"
${PYTHON:-python} scripts/challenge2/generate_synthetic_mast.py --output "$OUT" --target "$TARGET" --workers "$WORKERS" \
    --ranges "$RANGES" --r-target-range $RT --z-std "$ZSTD" --z-max "$ZMAX" --seed "${SYN_SEED:-2026}" $EXTRA
echo "== done"
