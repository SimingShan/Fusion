#!/bin/bash
# Stage candidate 20260921-070: equal-weight 5-member gated blend of existing checkpoints.
set -euo pipefail
cd "${FUSION_ROOT:?set FUSION_ROOT to the package root}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
${PYTHON:-python} -u scripts/optimization/finish_ensemble_candidate.py --candidate 20260921-070 \
  --members A=results/optimization/pod_scaled_li_v2/best.pt \
            B=results/optimization/pod_history128_li_v6/best.pt \
            B=results/training/c1_20260920_040/best.pt \
            B=results/training/c1_20260920_066/best.pt \
            B=results/training/c1_20260920_067/best.pt \
  --dev-predictions results/optimization/native_validation_v3/scaled_li/val_predictions.npz \
                    results/optimization/native_history128_li_v8/robust_history128_li/val_predictions.npz \
                    results/evaluations/20260920-040/native/val_predictions.npz \
                    results/evaluations/20260920-066/native/val_predictions.npz \
                    results/evaluations/20260920-067/native/val_predictions.npz \
  --legacy-predictions results/optimization/pod_scaled_li_v2/legacy_val_predictions.npz \
                       results/optimization/pod_history128_li_v6/legacy_val_predictions.npz \
                       results/training/c1_20260920_040/legacy_val_predictions.npz - - \
  --gate-threshold 0.1 --workers "${SLURM_CPUS_PER_TASK:-4}"
echo "=== candidate_evidence ==="
${PYTHON:-python} - <<'PY'
from pathlib import Path
from scripts.periodic_submission import candidate_evidence
try:
    print(candidate_evidence(Path("results/candidates/20260921-070")))
except Exception as exc:
    print("EVIDENCE_ERROR", type(exc).__name__, exc)
PY
