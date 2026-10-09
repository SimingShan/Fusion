#!/bin/bash
set -euo pipefail
cd "${FUSION_ROOT:?set FUSION_ROOT to the package root}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
${PYTHON:-python} scripts/challenge2/check_native_export_equivalence.py \
  --original results/challenge2/c2_export_smoke_013 --updated results/challenge2/c2_export_smoke_014 \
  --output results/challenge2/c2_export_016/export_equivalence.json
${PYTHON:-python} -u results/challenge2/c2_export_016/execution/distribute_frozen_export.py \
  --run results/challenge2/c2_export_016 --assemble
