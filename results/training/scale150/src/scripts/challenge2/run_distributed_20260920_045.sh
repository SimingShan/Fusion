#!/bin/bash
set -euo pipefail
cd "${FUSION_ROOT:?set FUSION_ROOT to the package root}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
${PYTHON:-python} - <<'PY'
import json, hashlib
from pathlib import Path
def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
run=Path('results/challenge2/c2_export_045')
gate=read(run/'promotion_plan.json')
assert sha(run/'export_plan.json')==gate['export_plan_sha256']
full=read(run/'export_plan.json')
for item in gate['bound_evidence']:
    assert sha(item['path'])==item['sha256'], item['path']
comparison=read(gate['development_comparison'])
assert comparison['same_queries'] and comparison['shots']==8 and comparison['frames']==1763
assert not comparison['mast_labels_read'] and not comparison['audit_labels_read']
assert comparison['scores']['trial_36']['S']-comparison['scores']['trial_29']['S']>=.0001
coverage=read(gate['development_coverage'])['machines']['DIII-D']
assert coverage['frames']==1763 and coverage['scalar_valid']>=1760 and not coverage['runtime_limited_frames']
coverage=read(gate['broad_coverage'])['machines']['MAST']
assert coverage['frames']==1642 and coverage['finite_positive_scalars']==1642
assert not coverage['runtime_limited_frames'] and not coverage['exceptions']
keys=('source_sha256','prior_sha256','packages','inventory_sha256','max_evaluations','max_seconds',
      'pressure_factor','allow_approximate_seed','retry_observed_pressure','solver_method','constrain_pressure_current')
for item in gate['native_export_checks']:
    path=Path(item['run']); plan=read(path/'export_plan.json'); result=read(path/'mast_export.json')
    assert sha(path/'export_plan.json')==item['plan_sha256']==result['plan_sha256']
    assert all(plan[k]==full[k] for k in keys)
    assert result['status']=='smoke_complete' and result['shots']==2 and result['frames']==item['frames']
    assert result['scalar_valid']==result['frames'] and result['runtime_limited_frames']==0
    assert sha(path/'mast_public_test.npz')==result['prediction_sha256']
    assert not result['test_labels_read'] and not result['audit_labels_read']
parity=read(gate['native_parity'])
assert parity['passed'] and parity['prediction_arrays_bit_identical'] and parity['shots']>=1
assert parity['export_plan_sha256']==gate['native_export_checks'][0]['plan_sha256']
assert not full['constrain_pressure_current'] and full['scope']=='full' and len(full['shots'])==1206
print(json.dumps({'complete_development_promotion_checks_passed':True,'submission_eligible':False}),flush=True)
PY
${PYTHON:-python} -u results/challenge2/c2_export_045/execution/distribute_frozen_export.py \
  --run results/challenge2/c2_export_045 --part "$SLURM_ARRAY_TASK_ID"
