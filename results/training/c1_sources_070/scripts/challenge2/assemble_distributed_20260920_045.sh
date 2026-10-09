#!/bin/bash
set -euo pipefail
cd "${FUSION_ROOT:?set FUSION_ROOT to the package root}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
${PYTHON:-python} -u results/challenge2/c2_export_045/execution/distribute_frozen_export.py \
  --run results/challenge2/c2_export_045 --assemble
${PYTHON:-python} scripts/challenge2/summarize_export_coverage.py \
  --run results/challenge2/c2_export_045 --output results/challenge2/c2_export_045/complete_coverage_review.json
