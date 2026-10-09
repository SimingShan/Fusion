#!/bin/bash
set -euo pipefail
cd "${FUSION_ROOT:?set FUSION_ROOT to the package root}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
task_snapshot=results/challenge2/c2_sources_005
mkdir -p "$task_snapshot/scripts/challenge2" "$task_snapshot/fusion"
cp scripts/challenge2/prepare_profile_prior.py scripts/challenge2/prepare_smooth_profiles.py scripts/challenge2/train_profile_prior.py scripts/challenge2/evaluate_profile_prior.py "$task_snapshot/scripts/challenge2/"
cp fusion/mast_profiles.py fusion/mast_physics.py fusion/smooth_profiles.py fusion/profile_scalars.py fusion/measured_pressure.py "$task_snapshot/fusion/"
${PYTHON:-python} -u scripts/challenge2/prepare_profile_prior.py \
  --train-shots 2048 --train-frames 4 --fresh-shots 16 --legacy-shots 8 --val-frames 4 \
  --output results/challenge2/c2_expanded_005
${PYTHON:-python} -u scripts/challenge2/prepare_smooth_profiles.py \
  --source results/challenge2/c2_expanded_005 --output results/challenge2/c2_targets_005
${PYTHON:-python} -u scripts/challenge2/train_profile_prior.py \
  --data results/challenge2/c2_targets_005 --output results/challenge2/c2_prior_005
${PYTHON:-python} -u scripts/challenge2/evaluate_profile_prior.py \
  --prior results/challenge2/c2_prior_005/prior.joblib --output results/challenge2/c2_evaluation_005 \
  --shots 8 --frames 3 --max-seconds 10 --measured-pressure --pressure-factor 2
${PYTHON:-python} -u scripts/challenge2/score_profile_trials.py \
  --trials 2 4 5 --output results/challenge2/development_comparison_003
