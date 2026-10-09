#!/usr/bin/env python3
"""Frozen, resumable native MAST prediction export with independent validation."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import importlib.metadata
import json
import multiprocessing
import os
import shutil
import sys
import zipfile

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
WORK=None


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n")
    temp.replace(path)


def clean(value):
    if isinstance(value,dict):return {k:clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [clean(v) for v in value]
    if isinstance(value,np.generic):return clean(value.item())
    if isinstance(value,float) and not np.isfinite(value):return None
    return value


def freeze(a):
    run=a.output.resolve()
    if (run/"export_plan.json").exists():
        raise ValueError("This export plan is already frozen")
    inventory_path=(ROOT/"data/raw/download_manifest.json").resolve()
    inventory=json.loads(inventory_path.read_text())
    shots=sorted((x for x in inventory["files"] if x["path"].startswith("data/mast_public_test/") and x["path"].endswith(".parquet")),key=lambda x:x["path"])
    if len(shots)!=1206 or len({x["path"] for x in shots})!=1206:
        raise ValueError("Pinned complete MAST inventory required")
    if a.smoke_shots:
        shots=shots[:a.smoke_shots]
    elif a.smoke_indices:
        if len(set(a.smoke_indices))!=len(a.smoke_indices) or min(a.smoke_indices)<0 or max(a.smoke_indices)>=len(shots):
            raise ValueError("Invalid declared smoke shot selection")
        shots=[shots[i] for i in sorted(a.smoke_indices)]
    run.mkdir(parents=True,exist_ok=True)
    snapshot=run/"source_snapshot"
    files=[Path(__file__),ROOT/"scripts/challenge2/evaluate_profile_prior.py",
           ROOT/"scripts/challenge2/prepare_profile_prior.py",ROOT/"scripts/codabench_submit.py",
           ROOT/"scripts/package_submission.py",ROOT/"starter/data_fixes.py"]
    files+=list((ROOT/"fusion").glob("*.py"))
    files+=list((ROOT/"starter/fusion_scoring").rglob("*.py"))
    files+=list((ROOT/"starter/fusion_scoring/masks").glob("*.npz"))
    files+=[p for p in (ROOT/"requirements.txt",ROOT/"uv.lock") if p.exists()]
    hashes={}
    for path in files:
        relative=path.resolve().relative_to(ROOT)
        target=snapshot/relative;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(path,target);hashes[str(relative)]=digest(target)
    frozen=run/"checkpoints/prior.joblib";frozen.parent.mkdir(exist_ok=True)
    expected=digest(a.prior);shutil.copyfile(a.prior,frozen)
    if digest(frozen)!=expected:
        raise ValueError("Prior changed while freezing")
    plan={"created_utc":datetime.now(timezone.utc).isoformat(),"scope":"smoke" if a.smoke_shots or a.smoke_indices else "full",
          "raw_root":str(inventory_path.parent),"inventory_sha256":digest(inventory_path),
          "prior_sha256":expected,"source_sha256":hashes,"shots":shots,
          "max_evaluations":600,"max_seconds":300,"pressure_factor":2.,
          "allow_approximate_seed":True,"expected_full_frames":69075,
          "retry_observed_pressure":bool(a.retry_observed_pressure),
          "retry_fresh_initialization":bool(a.retry_fresh_initialization),
          "solver_method":a.solver_method,"constrain_pressure_current":bool(a.constrain_pressure_current),
          "test_labels_read":False,"audit_labels_read":False,
          "packages":{key:importlib.metadata.version(key) for key in ("numpy","scipy","scikit-learn","joblib","pyarrow")}}
    write_json(run/"export_plan.json",plan)
    print(json.dumps({"frozen":True,"scope":plan["scope"],"shots":len(shots),"run":str(run)}),flush=True)


def initialize(run):
    global WORK
    import joblib
    run=Path(run)
    if ROOT!=(run/"source_snapshot").resolve():
        raise ValueError("Execute the frozen script from its source snapshot")
    plan=json.loads((run/"export_plan.json").read_text())
    if digest(Path(plan["raw_root"])/"download_manifest.json")!=plan["inventory_sha256"]:
        raise ValueError("Pinned inventory changed")
    for name,expected in plan["source_sha256"].items():
        if digest(ROOT/name)!=expected:
            raise ValueError("Frozen source changed")
    for key,version in plan["packages"].items():
        if importlib.metadata.version(key)!=version:
            raise ValueError("Scientific dependency version changed")
    prior_path=run/"checkpoints/prior.joblib"
    if digest(prior_path)!=plan["prior_sha256"]:
        raise ValueError("Frozen prior changed")
    prior=joblib.load(prior_path)
    for key in ("model","coil_model"):
        if "n_jobs" in prior[key].get_params(deep=False):
            prior[key].set_params(n_jobs=1)
    from scripts.challenge2.evaluate_profile_prior import infer,infer_with_retry,infer_with_restart
    if plan.get("retry_observed_pressure"):
        infer=infer_with_retry
    if plan.get("retry_fresh_initialization"):
        if not plan.get("retry_observed_pressure"):
            raise ValueError("Fresh initialization requires the existing conditional pressure policy")
        infer=infer_with_restart
    from fusion.mast_profiles import read_row,FreeBoundaryGS
    with np.load(ROOT/"starter/fusion_scoring/masks/mast_envelope.npz",allow_pickle=False) as z:
        envelope=z["mask_coarse"].astype(bool)
    WORK={"run":run,"plan":plan,"plan_sha256":digest(run/"export_plan.json"),"prior":prior,
          "infer":infer,"read_row":read_row,"solver_type":FreeBoundaryGS,"solver":None,"envelope":envelope}


def predict_shot(index):
    w=WORK;run,plan=w["run"],w["plan"];item=plan["shots"][index]
    path=Path(plan["raw_root"])/item["path"]
    output=run/"shots"/f"{index:04d}.npz";receipt=output.with_suffix(".json")
    if receipt.exists():
        record=json.loads(receipt.read_text())
        if record.get("status")=="complete":
            if record["plan_sha256"]!=w["plan_sha256"] or digest(output)!=record["prediction_sha256"]:
                raise ValueError("A completed shot or its plan changed")
            return {k:record[k] for k in ("index","frames","converged","scalar_valid","runtime_limited_frames","maximum_negative_mass_fraction","observed_pressure_retry_frames","initialization_retry_frames")}
    if not path.is_file() or path.stat().st_size!=item["size"]:
        raise ValueError("Missing or incomplete pinned input")
    row=w["read_row"](path,labels=False)
    if row["source"]!="MAST":
        raise ValueError("Wrong input machine")
    if w["solver"] is None:
        w["solver"]=w["solver_type"](row["efit_grid_R"],row["efit_grid_Z"])
    solver=w["solver"]
    if not (np.array_equal(solver.R,row["efit_grid_R"]) and np.array_equal(solver.Z,row["efit_grid_Z"])):
        raise ValueError("Input grid changed")
    times=np.asarray(row["efit_times"],float)
    if not np.isfinite(times).all() or np.any(np.diff(times)<=0):
        raise ValueError("Invalid native time ordering")
    n=len(times);fields=np.empty((n,65,65),np.float32);q=np.empty(n,np.float32);beta=np.empty(n,np.float32)
    previous=None;records=[]
    keys=("converged","status","nonlinear_profile_relative_residual","function_evaluations",
          "solver_elapsed_seconds","input_Ip_A","Ip_relative_error","current_negative_mass_fraction",
          "scalar_valid","pressure_relative_rmse","pressure_fit_used","positive_pressure_outside_surface_fraction",
          "pressure_axis_Pa","q95","betaN","pressure_observation_retry_used","original_prediction_failure",
          "pressure_current_bound_active","unconstrained_pressure_current_A","constrained_pressure_current_A",
          "measured_current_constraint_A","pressure_prior_amplitude_constrained",
          "initialization_retry_used","original_continuation_failure","initialization_retry_failure")
    try:
        for frame in range(n):
            field,current,info=w["infer"](row,frame,w["prior"],solver,w["envelope"],previous,
                plan["max_seconds"],plan["max_evaluations"],True,plan["pressure_factor"],
                solver_method=plan.get("solver_method","krylov"),
                constrain_pressure_current=plan.get("constrain_pressure_current",False))
            if not info.get("scalar_valid") or not np.isfinite(field).all():
                raise ValueError(f"Invalid prediction at native frame {frame}")
            scalar=np.array([info["q95"],info["betaN"]],float)
            if not np.isfinite(scalar).all() or np.any(scalar<=0):
                raise ValueError(f"Nonpositive or nonfinite scalar at native frame {frame}")
            fields[frame],q[frame],beta[frame]=field,scalar[0],scalar[1]
            if info["converged"]:
                previous=current
            elif plan["allow_approximate_seed"] and info.get("nonlinear_profile_relative_residual") is not None:
                if info["nonlinear_profile_relative_residual"]<.02 and np.min(current)>=-1e-6:
                    previous=current
            records.append(clean({"frame":frame,"time_ms":float(times[frame]),**{k:info.get(k) for k in keys}}))
    except Exception as exc:
        write_json(receipt,{"status":"failed","index":index,"raw_path":item["path"],
                           "plan_sha256":w["plan_sha256"],"failure":f"{type(exc).__name__}: {exc}","completed_frames":len(records)})
        raise
    output.parent.mkdir(parents=True,exist_ok=True)
    temporary=output.with_name(output.stem+f".{os.getpid()}.tmp.npz")
    np.savez_compressed(temporary,psirz=fields,q95=q,betaN=beta)
    temporary.replace(output)
    record={"status":"complete","index":index,"raw_path":item["path"],"frames":n,
            "native_times_sha256":hashlib.sha256(np.asarray(times,dtype="<f8").tobytes()).hexdigest(),
            "prediction_sha256":digest(output),"plan_sha256":w["plan_sha256"],
            "converged":sum(x["converged"] for x in records),"scalar_valid":n,
            "runtime_limited_frames":sum(x["solver_elapsed_seconds"]>=plan["max_seconds"] for x in records),
            "maximum_negative_mass_fraction":max(x["current_negative_mass_fraction"] for x in records),
            "observed_pressure_retry_frames":sum(bool(x.get("pressure_observation_retry_used")) for x in records),
            "initialization_retry_frames":sum(bool(x.get("initialization_retry_used")) for x in records),
            "records":records}
    write_json(receipt,record)
    return {k:record[k] for k in ("index","frames","converged","scalar_valid","runtime_limited_frames","maximum_negative_mass_fraction","observed_pressure_retry_frames","initialization_retry_frames")}


def assemble(run,plan,summaries):
    output=run/"mast_public_test.npz";temp=output.with_suffix(".tmp.npz")
    with zipfile.ZipFile(temp,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as z:
        for i in range(len(plan["shots"])):
            receipt=json.loads((run/"shots"/f"{i:04d}.json").read_text())
            shot=run/"shots"/f"{i:04d}.npz"
            if receipt["status"]!="complete" or digest(shot)!=receipt["prediction_sha256"]:
                raise ValueError("Incomplete or changed shot")
            with np.load(shot,allow_pickle=False) as arrays:
                for key in ("psirz","q95","betaN"):
                    with z.open(f"shot_{i:04d}_{key}.npy","w",force_zip64=True) as stream:
                        np.lib.format.write_array(stream,arrays[key],allow_pickle=False)
    temp.replace(output)
    raw=Path(plan["raw_root"])
    if plan["scope"]=="smoke":
        manifest=json.loads((raw/"download_manifest.json").read_text())
        manifest["files"]=plan["shots"]
        raw=run/"smoke_raw";raw.mkdir(exist_ok=True)
        write_json(raw/"download_manifest.json",manifest)
        for item in plan["shots"]:
            destination=raw/item["path"];destination.parent.mkdir(parents=True,exist_ok=True)
            if not destination.exists():destination.symlink_to(Path(plan["raw_root"])/item["path"])
    from scripts.package_submission import validate_prediction
    validation=validate_prediction(output,raw,len(plan["shots"]),"mast_public_test")
    for shot in validation["shots"]:
        receipt=json.loads((run/"shots"/f"{shot['prediction_index']:04d}.json").read_text())
        if shot["raw_path"]!=receipt["raw_path"] or shot["efit_times_sha256"]!=receipt["native_times_sha256"]:
            raise ValueError("Native shot or timestamp ordering changed")
    if plan["scope"]=="full" and (validation["frames"]!=69075 or validation["validated_shots"]!=1206):
        raise ValueError("Incomplete full-fold export")
    write_json(run/"structural_validation.json",validation)
    report={"status":"complete" if plan["scope"]=="full" else "smoke_complete","scope":plan["scope"],
            "prediction":str(output),"prediction_sha256":digest(output),
            "prior_sha256":plan["prior_sha256"],"plan_sha256":digest(run/"export_plan.json"),
            "source_sha256":plan["source_sha256"],"shots":len(summaries),
            "frames":sum(x["frames"] for x in summaries),"converged":sum(x["converged"] for x in summaries),
            "scalar_valid":sum(x["scalar_valid"] for x in summaries),
            "runtime_limited_frames":sum(x["runtime_limited_frames"] for x in summaries),
            "maximum_negative_mass_fraction":max(x["maximum_negative_mass_fraction"] for x in summaries),
            "observed_pressure_retry_frames":sum(x.get("observed_pressure_retry_frames",0) for x in summaries),
            "initialization_retry_frames":sum(x.get("initialization_retry_frames",0) for x in summaries),
            "test_labels_read":False,"audit_labels_read":False,"eligible_for_submission":False,
            "limitations":"Numerical convergence is distinct from unverified MAST accuracy; separate transfer review required"}
    write_json(run/"mast_export.json",report)
    print(json.dumps({k:v for k,v in report.items() if k not in ("source_sha256",)}),flush=True)


def execute(a):
    run=a.output.resolve()
    initialize(run)
    plan=WORK["plan"];summaries=[];failures=[]
    with ProcessPoolExecutor(max_workers=a.workers,mp_context=multiprocessing.get_context("spawn"),
                             initializer=initialize,initargs=(str(run),)) as pool:
        futures={pool.submit(predict_shot,i):i for i in range(len(plan["shots"]))}
        for future in as_completed(futures):
            index=futures[future]
            try:summaries.append(future.result())
            except Exception as exc:failures.append({"index":index,"error":f"{type(exc).__name__}: {exc}"})
            write_json(run/"progress.json",{"completed_shots":len(summaries),"failures":failures,
                "completed_frames":sum(x["frames"] for x in summaries),"total_shots":len(plan["shots"]),
                "converged_frames":sum(x["converged"] for x in summaries),"summaries":summaries})
            print(json.dumps({"completed_shots":len(summaries),"failed_shots":len(failures),"total_shots":len(plan["shots"])}),flush=True)
    if failures:
        raise RuntimeError("Some shots failed; no complete prediction archive is eligible")
    assemble(run,plan,summaries)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--prior",type=Path,default=Path("results/challenge2/c2_prior_002/prior.joblib"))
    p.add_argument("--freeze",action="store_true")
    p.add_argument("--smoke-shots",type=int,default=0)
    p.add_argument("--smoke-indices",type=int,nargs="*")
    p.add_argument("--retry-observed-pressure",action="store_true")
    p.add_argument("--retry-fresh-initialization",action="store_true")
    p.add_argument("--solver-method",choices=("krylov","anderson","picard"),default="krylov")
    p.add_argument("--constrain-pressure-current",action="store_true")
    p.add_argument("--workers",type=int,default=1)
    a=p.parse_args()
    if a.workers<1 or not 0<=a.smoke_shots<1206:p.error("Invalid worker or smoke count")
    if a.smoke_shots and a.smoke_indices:p.error("Declare only one smoke selection")
    if a.retry_fresh_initialization and not a.retry_observed_pressure:p.error("Fresh initialization requires the existing conditional pressure policy")
    if a.freeze:freeze(a)
    else:execute(a)


if __name__=="__main__":
    main()
