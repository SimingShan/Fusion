#!/usr/bin/env python3
"""Run one authorized audit of the already frozen v5 prediction ensemble.

Register the protocol and copy/hash the exact exported checkpoints before any
audit label access. Then prepare every native frame of the original 128 audit
shots, infer only the combined fixed candidate, and score it exactly once.
No training, alternative predictions, threshold search, or expert scores occur.
"""
from __future__ import annotations

import argparse
import ast
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
OUTPUT=ROOT/"results/optimization/audit_frozen_v5"
BASE=ROOT/"data/advanced_audit_native_v5"
HISTORY=ROOT/"data/advanced_history_audit_native_v5"


def utc():return datetime.now(timezone.utc).isoformat()


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b""):digest.update(chunk)
    return digest.hexdigest()


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n")
    os.replace(temporary,path)


def inference_ast(path):
    tree=ast.parse(Path(path).read_text())
    selected=[node for node in tree.body if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef))
              and node.name in ("reconstruct","torch_normalization")]
    if len(selected)!=2:raise ValueError("Both inference helpers are required")
    return hashlib.sha256(ast.dump(ast.Module(body=selected,type_ignores=[]),include_attributes=False).encode()).hexdigest()


def register():
    path=OUTPUT/"protocol.json"
    if path.exists():
        protocol=json.loads(path.read_text());verify(protocol);return protocol
    source=ROOT/"results/optimization/submission_scaled_robust_v5/export.json"
    exported=json.loads(source.read_text())
    if exported.get("status")!="complete":raise ValueError("The submitted candidate must already be fully exported")
    manifest_path=ROOT/"data/advanced_v1/manifest.json"
    manifest=json.loads(manifest_path.read_text());splits=manifest.get("splits",manifest)
    audit=splits["audit"]
    identities=[{k:entry[k] for k in ("shot_index","shot_id","shard","row","raw_frames")} for entry in audit]
    selected={entry["shot_index"] for entry in audit}
    if len(selected)!=128:raise ValueError("The registered reserved audit must have exactly 128 unique shots")
    if any(selected & {entry["shot_index"] for entry in splits[name]} for name in ("train","val","legacy_val")):
        raise ValueError("Audit overlaps training or development shots")
    code=dict(exported["source_code_sha256"])
    original_helper=ROOT/"results/optimization/provenance/train_pod_v5_export.py"
    expected_helper=code.pop("scripts/train_pod.py")
    if sha(original_helper)!=expected_helper:raise ValueError("Preserved pre-edit training source differs from export hash")
    if inference_ast(original_helper)!=inference_ast(ROOT/"scripts/train_pod.py"):
        raise ValueError("Current inference helpers differ from the exact exported source")
    frozen_helper=OUTPUT/"frozen/train_pod_v5_export.py";frozen_helper.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(original_helper,frozen_helper)
    for relative in ("fusion/advanced_data.py","fusion/data.py","starter/data_fixes.py",
                     "scripts/optimization/prepare_history_features.py","scripts/evaluate.py"):
        code[relative]=sha(ROOT/relative)
    for relative,expected in code.items():
        if sha(ROOT/relative)!=expected:raise ValueError(f"Source changed since the official export: {relative}")
    frozen={}
    for role in ("primary","secondary"):
        original=Path(exported[role]);expected=exported["source_checkpoint_sha256"][role]
        if sha(original)!=expected:raise ValueError(f"Frozen {role} differs from the submitted candidate")
        target=OUTPUT/"frozen"/f"{role}.pt";target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(original,target)
        if sha(target)!=expected:raise ValueError("Checkpoint copy failed verification")
        frozen[role]={"source":str(original),"path":str(target),"sha256":expected}
    protocol={"registered_utc":utc(),"authorization":"Root explicitly authorized one-time evaluation of the frozen v5 candidate on all native frames of the reserved 128 audit shots; no tuning or multiple candidate comparisons.",
              "candidate":"scaled-li POD + robust-history POD; fixed 75/25 with input-quality fallback",
              "source_export":str(source),"source_export_sha256":sha(source),"frozen_checkpoints":frozen,
              "primary_weight":.75,"gate_threshold":.1,
              "gate_rule":exported["gate_rule"],"secondary_expert":"robust-history POD, not currents-only",
              "source_manifest":str(manifest_path),"source_manifest_sha256":sha(manifest_path),
              "dataset":manifest["dataset"],"audit_shots":identities,"expected_shots":128,
              "expected_native_frames":sum(entry["raw_frames"] for entry in audit),
              "audit_identities_sha256":hashlib.sha256(json.dumps(identities,sort_keys=True).encode()).hexdigest(),
              "source_code_sha256":code,"wrapper_sha256":sha(Path(__file__)),
              "inference_helper_proof":{"original_source":str(frozen_helper),"original_sha256":expected_helper,
                  "helpers_ast_sha256":inference_ast(frozen_helper),"current_source":"scripts/train_pod.py",
                  "qualification":"The shared training script subsequently gained optional training-only geometry loss. The exact pre-edit source was recovered and SHA256 matches v5 export. reconstruct and torch_normalization have identical ASTs; inference is unchanged."},
              "audit_labels_read_during_registration":False,
              "prior_preparation_note":"The original capped audit cache was prepared earlier, but audit labels were not used in fitting, checkpoint selection, or evaluation. This records first authorized audit evaluation consumption, not first-ever physical disk access.",
              "primary_endpoint":"Unmodified official scorer composite S on all native frames of the original 128 audit shots",
              "secondary_reports":["official metric components and derivation failure fractions","input-only gate counts"],
              "prohibited_uses":["training or normalization fitting","checkpoint or blend selection","threshold/weight search","per-shot target corrections","alternative candidate scoring"],
              "analysis":"Report the single score and a descriptive gap from the already recorded development results; do not tune from the audit or treat a negative gap as a gate for changing this submitted candidate.",
              "repeat_policy":"Technical resumption is allowed only with identical frozen candidate and selection. A completed score is never recomputed by this script."}
    write(path,protocol)
    write(OUTPUT/"events.json",{"registered_utc":protocol["registered_utc"],"status":"registered_before_audit_label_access"})
    return protocol


def verify(protocol):
    for info in protocol["frozen_checkpoints"].values():
        if sha(info["path"])!=info["sha256"]:raise ValueError("Frozen checkpoint changed")
    for relative,expected in protocol["source_code_sha256"].items():
        if sha(ROOT/relative)!=expected:raise ValueError(f"Pinned source changed: {relative}")
    proof=protocol["inference_helper_proof"]
    if sha(proof["original_source"])!=proof["original_sha256"] or inference_ast(ROOT/proof["current_source"])!=proof["helpers_ast_sha256"]:
        raise ValueError("Exported inference helper equivalence no longer holds")


def run(protocol,workers):
    verify(protocol)
    if (OUTPUT/"audit_score.json").exists():
        print("Completed audit exists; refusing to evaluate again.",flush=True);return
    from fusion.advanced_data import prepare_advanced_dataset
    from scripts.optimization.prepare_history_features import _task,history_feature_names
    from scripts.optimization.export_gated_ensemble import export
    from scripts.evaluate import scorer_fingerprint
    # Plan-only resolves the deterministic metadata selection without shot arrays.
    planned=prepare_advanced_dataset(output=BASE,val_frames=0,splits=["audit"],plan_only=True)
    expected=[entry["shot_index"] for entry in protocol["audit_shots"]]
    if planned["selection"]["audit"]!=expected:raise ValueError("Regenerated audit identities changed")
    events=json.loads((OUTPUT/"events.json").read_text())
    events.setdefault("first_audit_label_consumption_start_utc",utc())
    events.update(status="preparing_audit_native_targets",scorer_sha256=scorer_fingerprint(),
                  note="Timestamp written immediately before invoking audit cache preparation; target reads start in its workers.")
    write(OUTPUT/"events.json",events)
    base=prepare_advanced_dataset(output=BASE,val_frames=0,splits=["audit"],workers=min(4,workers),consolidate=False)
    native=[{k:entry[k] for k in ("shot_index","shot_id","shard","row","raw_frames")} for entry in base["audit"]]
    if native!=protocol["audit_shots"] or sum(e["n_frames"] for e in base["audit"])!=protocol["expected_native_frames"]:
        raise ValueError("Prepared audit differs from the preregistered complete native selection")
    history=json.loads(json.dumps(base));history["feature_names"]+=history_feature_names()
    history["feature_augmentation"]={"base_dir":str(BASE),"base_manifest_sha256":sha(BASE/"manifest.json"),
        "target_values_read":False,"authorized_audit_input_view":True,"complete_splits":{"audit":True}}
    history["consolidated"]={}
    tasks=[]
    for entry in history["audit"]:
        entry["path"]=str(BASE/entry["path"])
        entry["X_extra"]=f"history/audit/shot_{entry['shot_index']:05d}.npy"
        tasks.append({"output":str(HISTORY),"raw_dir":str(ROOT/"data/raw"),"split":"audit","entry":entry})
    with ProcessPoolExecutor(max_workers=min(4,workers)) as executor:
        list(executor.map(_task,tasks))
    write(HISTORY/"manifest.json",history)
    events.update(status="exporting_only_frozen_ensemble",input_cache_complete_utc=utc())
    write(OUTPUT/"events.json",events)
    prediction=export(SimpleNamespace(primary=Path(protocol["frozen_checkpoints"]["primary"]["path"]),
        secondary=Path(protocol["frozen_checkpoints"]["secondary"]["path"]),primary_data=BASE,
        secondary_data=HISTORY,output=OUTPUT,split="audit",batch_size=512,workers=workers,device="cpu"))
    verify(protocol)
    events.update(status="scoring_frozen_candidate_once",prediction_complete_utc=utc(),prediction_sha256=sha(prediction))
    write(OUTPUT/"events.json",events)
    subprocess.run([sys.executable,str(ROOT/"scripts/evaluate.py"),"--data-dir",str(BASE),"--split","audit",
                    "--pred",str(prediction),"--output",str(OUTPUT/"audit_score.json"),"--workers",str(workers)],check=True)
    score=json.loads((OUTPUT/"audit_score.json").read_text())
    score["evaluation"]["type"]="one_time_frozen_model_audit"
    score["audit"]={"protocol":str(OUTPUT/"protocol.json"),"protocol_sha256":sha(OUTPUT/"protocol.json"),
        "audit_labels_read":True,"used_for_model_selection":False,"candidate_count":1,
        "source_export_sha256":protocol["source_export_sha256"],"first_consumption_utc":events["first_audit_label_consumption_start_utc"]}
    write(OUTPUT/"audit_score.json",score)
    verify(protocol)
    events.update(status="complete",score_written_utc=utc(),composite_S=score["S"],
                  evaluated_shots=score["evaluation"]["evaluated_shots"],evaluated_frames=score["evaluation"]["evaluated_frames"])
    write(OUTPUT/"events.json",events)
    print(json.dumps(events,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--register-only",action="store_true")
    parser.add_argument("--workers",type=int,default=8)
    args=parser.parse_args();os.chdir(ROOT)
    if args.workers<1:parser.error("workers must be positive")
    protocol=register()
    if args.register_only:
        print(json.dumps({k:protocol[k] for k in ("registered_utc","expected_shots","expected_native_frames")}),flush=True)
    else:run(protocol,args.workers)


if __name__=="__main__":main()
