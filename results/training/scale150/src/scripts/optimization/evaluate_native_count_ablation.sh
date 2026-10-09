#!/bin/bash
set -euo pipefail
task_workers="${SLURM_CPUS_PER_TASK:-8}"
export OMP_NUM_THREADS="$task_workers" OPENBLAS_NUM_THREADS="$task_workers" MKL_NUM_THREADS="$task_workers"
export CUDA_VISIBLE_DEVICES=""
${PYTHON:-python} -u scripts/optimization/reuse_identical_count_evaluation.py
for task_arm in control weighted; do
    task_model_dir="results/optimization/pod_history128_count_${task_arm}_v5"
    task_result_dir="results/optimization/native_count_ablation_v5/${task_arm}"
    if test -f "$task_result_dir/reused_identical_model.json"; then
        continue
    fi
    ${PYTHON:-python} -u scripts/optimization/gated_fourier_blend.py --data-dir data/advanced_v1 --primary results/optimization/pod_scaled_li_v2 --expert-name scaled-li-POD-DeepONet --secondary "$task_model_dir" --secondary-name "robust-history128-count-${task_arm}-POD-DeepONet" --output "$task_result_dir/capped" --workers "$task_workers" --skip-test-coverage
    mkdir -p "$task_result_dir/native"
    cp "$task_model_dir/best.pt" "$task_result_dir/native/best.pt"
    ${PYTHON:-python} -u scripts/train_pod.py --data-dir data/advanced_history_val_native_v3 --output "$task_result_dir/native" --predict-only --split val --workers "$task_workers" --batch-size 512
    ${PYTHON:-python} -u scripts/optimization/gated_fourier_blend.py --data-dir data/advanced_val_native_v3 --primary results/optimization/native_validation_v3/scaled_li --expert-name scaled-li-POD-DeepONet --secondary "$task_result_dir/native" --secondary-name "robust-history128-count-${task_arm}-POD-DeepONet" --output "$task_result_dir/native_combined" --workers "$task_workers" --splits val --skip-test-coverage
done
${PYTHON:-python} - <<'PY'
from pathlib import Path
import hashlib,json,torch
base=Path('results/optimization/native_count_ablation_v5')
report={'evaluation_type':'matched_native_count_loss_weighting_ablation','audit_labels_read':False,
        'official_submission':False,'init_sha256':hashlib.sha256((base/'init.pt').read_bytes()).hexdigest(),'arms':{}}
runs={}
for arm in ('control','weighted'):
    model=Path(f'results/optimization/pod_history128_count_{arm}_v5')
    run=json.loads((model/'run.json').read_text());runs[arm]=run
    result={'completed_epochs':run['completed_epochs'],'selected_epoch':torch.load(model/'best.pt',map_location='cpu',weights_only=False)['epoch']}
    for split in ('val','legacy_val'):
        result[f'{split}_expert_S']=json.loads((model/f'{split}_score.json').read_text())['S']
    result['native_expert_S']=json.loads((base/arm/'native/val_score.json').read_text())['S']
    capped=json.loads((base/arm/'capped/summary.json').read_text())
    result['capped_fixed_blend_S']={split:value['scores']['gated_blend75']['S'] for split,value in capped['splits'].items()}
    native=json.loads((base/arm/'native_combined/summary.json').read_text())
    result['native_fixed_blend_S']=native['splits']['val']['scores']['gated_blend75']['S']
    result['frame_weighting']=run.get('frame_weighting')
    reuse=base/arm/'reused_identical_model.json'
    result['reused_identical_predictor_evaluation']=json.loads(reuse.read_text()) if reuse.exists() else None
    report['arms'][arm]=result
control=runs['control']['arguments'];weighted=runs['weighted']['arguments']
differences={k:[control[k],weighted[k]] for k in control if control[k]!=weighted[k]}
assert set(differences)=={'output','frame_weighting'},differences
assert runs['control']['completed_epochs']==runs['weighted']['completed_epochs']==100
report['argument_differences']=differences
report['matched_arguments']={k:v for k,v in control.items() if k not in differences}
(base/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report['arms'],indent=2))
PY
