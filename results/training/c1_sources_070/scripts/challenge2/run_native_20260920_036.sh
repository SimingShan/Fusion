#!/bin/bash
set -euo pipefail
cd "${FUSION_ROOT:?set FUSION_ROOT to the package root}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
${PYTHON:-python} -u results/challenge2/c2_sources_033/scripts/challenge2/evaluate_profile_prior.py \
  --prior results/challenge2/c2_prior_002/prior.joblib \
  --data-root ${FUSION_ROOT}/data \
  --output "results/challenge2/c2_native_development_036/part_${SLURM_ARRAY_TASK_ID}" \
  --shots 8 --frames 0 --machines DIII-D --max-seconds 300 --max-evaluations 600 \
  --measured-pressure --pressure-factor 2 --allow-approximate-seed --retry-observed-pressure \
  --solver-method anderson --prediction-jobs 1 --shard-count 4 --shard-index "$SLURM_ARRAY_TASK_ID"
