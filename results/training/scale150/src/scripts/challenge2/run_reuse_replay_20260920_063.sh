#!/bin/bash
set -euo pipefail
cd "${FUSION_ROOT:?set FUSION_ROOT to the package root}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
task_run=results/challenge2/c2_export_smoke_063
${PYTHON:-python} scripts/challenge2/reuse_restart_export.py prepare --run "$task_run" \
  --parent results/challenge2/c2_export_smoke_046 \
  --parity-original results/challenge2/c2_export_smoke_044 \
  --parity-updated results/challenge2/c2_export_smoke_057 \
  --correction results/challenge2/c2_export_smoke_058
${PYTHON:-python} "$task_run/reuse/source_snapshot/scripts/challenge2/reuse_restart_export.py" adopt --run "$task_run"
${PYTHON:-python} "$task_run/source_snapshot/scripts/challenge2/export_mast_predictions.py" --output "$task_run" --workers 1
${PYTHON:-python} scripts/challenge2/check_native_export_equivalence.py \
  --original results/challenge2/c2_export_smoke_046 --updated "$task_run" \
  --output results/checks/c2_restart_replay_063.json
${PYTHON:-python} - <<'PY'
from pathlib import Path
import json
from scripts.codabench_submit import write_json
run=Path('results/challenge2/c2_export_smoke_063')
report=json.loads((run/'mast_export.json').read_text())
original=json.loads(Path('results/challenge2/c2_export_smoke_046/mast_export.json').read_text())
assert report['frames']==report['scalar_valid']==153
assert report['prediction_sha256']==original['prediction_sha256']
assert report['observed_pressure_retry_frames']==original['observed_pressure_retry_frames']==2
assert report['initialization_retry_frames']==0
records=[json.loads(p.read_text()) for p in (run/'shots').glob('*.json')]
assert len(records)==2 and all(r['reuse']['prediction_recomputed'] is False for r in records)
path=Path('results/checks/c2_restart_replay_063.json');check=json.loads(path.read_text())
check.update(frozen_runner_and_cached_export_passed=True,reused_shots=2,reused_frames=153,
             existing_pressure_retry_frames_preserved=2,prediction_arrays_bit_identical=True)
write_json(path,check)
print('Frozen replay passed: 153 frames, both existing conditional retries preserved',flush=True)
PY
