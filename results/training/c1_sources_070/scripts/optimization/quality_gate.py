#!/usr/bin/env python3
"""Evaluate a predeclared input-quality gate between two frozen POD experts.

Uses full-POD predictions when >10% of real edge Thomson channels have positive,
observed Te AND ne; otherwise uses the currents-only expert. Also evaluates the
fixed 0.75 full + 0.25 currents blend. No threshold search, target correction, or
shot-identity condition is permitted in the gate. Audit data are never loaded.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
THRESHOLD = 0.10
BLEND_FULL_WEIGHT = 0.75
SPLITS = ("val", "legacy_val")
FIELDS = ("Te_log1p", "Te_observed", "ne_log1p", "ne_observed", "geometry_observed")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def feature_columns(names):
    result, channels = {}, None
    for field in FIELDS:
        prefix = f"thomson_edge_{field}_channel_"
        items = sorted((int(name[len(prefix):]), i) for i, name in enumerate(names) if name.startswith(prefix))
        if not items:
            raise ValueError(f"Required gate features missing: {prefix}")
        current_channels = [item[0] for item in items]
        if channels is not None and current_channels != channels:
            raise ValueError("Gate feature channel identities do not align")
        channels = current_channels
        result[field] = np.asarray([item[1] for item in items], dtype=np.int64)
    return result


def gate_from_inputs(X, columns):
    """Pure input-only rule; no targets, predictions, or shot IDs are arguments."""
    values = {name: X[:, index] for name, index in columns.items()}
    if not all(np.isfinite(value).all() for value in values.values()):
        raise ValueError("Nonfinite cached gate inputs")
    observed_te = values["Te_observed"] > 0
    observed_ne = values["ne_observed"] > 0
    # Geometry identifies present channels even when diagnostic values are absent.
    # Observation masks provide a fallback for a real channel lacking geometry.
    # Padding has neither geometry nor observed measurements and is excluded.
    real = (values["geometry_observed"] > 0) | observed_te | observed_ne
    useful = real & observed_te & observed_ne & (values["Te_log1p"] > 0) & (values["ne_log1p"] > 0)
    real_count, useful_count = real.sum(1), useful.sum(1)
    fraction = np.divide(useful_count, real_count, out=np.zeros(len(X), dtype=np.float64), where=real_count>0)
    return fraction <= THRESHOLD, fraction, real_count, useful_count


def write_array(archive, key, array):
    array = np.asarray(array, dtype=np.float32)
    if not np.isfinite(array).all():
        raise ValueError(f"Nonfinite prediction: {key}")
    with archive.open(key+".npy", "w", force_zip64=True) as stream:
        np.lib.format.write_array(stream, array, allow_pickle=False)


def build_predictions(data_dir, entries, names, source_full, source_currents, output, split):
    columns = feature_columns(names)
    output_paths = {method: output/method/f"{split}_predictions.npz" for method in ("gate", "blend75")}
    for path in output_paths.values():
        path.parent.mkdir(parents=True,exist_ok=True)
    archives = {name: zipfile.ZipFile(path.with_suffix(".tmp.npz"),"w",compression=zipfile.ZIP_DEFLATED,
                                    compresslevel=1,allowZip64=True) for name,path in output_paths.items()}
    report = {"shots":len(entries), "frames":0, "gated_frames":0, "gated_shots":0,
              "real_channels_min":None, "real_channels_max":None, "per_shot":[]}
    try:
        with np.load(source_full,allow_pickle=False) as full, np.load(source_currents,allow_pickle=False) as currents:
            expected={f"shot_{i:04d}_{name}" for i in range(len(entries)) for name in ("psirz","q95","betaN")}
            if set(full.files)!=expected or set(currents.files)!=expected:
                raise ValueError("Expert prediction keys do not match the requested complete validation fold")
            for i,entry in enumerate(entries):
                with np.load(data_dir/entry["path"],allow_pickle=False) as shot:
                    # Deliberately read only X for gating; labels enter evaluation later.
                    X=shot["X"]
                selected,fraction,real,useful=gate_from_inputs(X,columns)
                n=len(X)
                if n!=entry["n_frames"]:
                    raise ValueError("Cache frame count disagrees with split manifest")
                for name in ("psirz","q95","betaN"):
                    key=f"shot_{i:04d}_{name}"
                    A,B=full[key],currents[key]
                    shape=(n,65,65) if name=="psirz" else (n,)
                    if A.shape!=shape or B.shape!=shape or not np.isfinite(A).all() or not np.isfinite(B).all():
                        raise ValueError(f"Invalid expert prediction for {key}")
                    mask=selected[:,None,None] if name=="psirz" else selected
                    gated=np.where(mask,B,A)
                    blended=BLEND_FULL_WEIGHT*A+(1-BLEND_FULL_WEIGHT)*B
                    write_array(archives["gate"],key,gated)
                    write_array(archives["blend75"],key,blended)
                report["frames"]+=n
                report["gated_frames"]+=int(selected.sum())
                report["gated_shots"]+=int(selected.any())
                low,high=int(real.min()),int(real.max())
                report["real_channels_min"]=low if report["real_channels_min"] is None else min(low,report["real_channels_min"])
                report["real_channels_max"]=high if report["real_channels_max"] is None else max(high,report["real_channels_max"])
                report["per_shot"].append({"validation_index":i,"shot_index":entry.get("shot_index"),
                    "shot_id":entry.get("shot_id"),"frames":n,"gated_frames":int(selected.sum()),
                    "gate_frame_indices":np.flatnonzero(selected).tolist(),
                    "useful_fraction_min":float(fraction.min()),"useful_fraction_max":float(fraction.max()),
                    "real_channels_min":low,"real_channels_max":high,
                    "useful_channels_min":int(useful.min()),"useful_channels_max":int(useful.max())})
    finally:
        for archive in archives.values():
            archive.close()
    for path in output_paths.values():
        os.replace(path.with_suffix(".tmp.npz"),path)
    return output_paths,report


def evaluate(data_dir,split,pred,score_path,workers,policy_hash,method):
    score_path.parent.mkdir(parents=True,exist_ok=True)
    command=[sys.executable,str(ROOT/"scripts/evaluate.py"),"--data-dir",str(data_dir),"--split",split,
             "--pred",str(pred),"--output",str(score_path),"--workers",str(workers)]
    with score_path.with_suffix(".log").open("w") as stream:
        subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,check=True)
    score=json.loads(score_path.read_text())
    score["experiment"]={"type":"predeclared_input_quality_gate_comparison","method":method,
                         "policy_sha256":policy_hash,"audit_labels_read":False,
                         "threshold_searched":False,"model_prediction":True,
                         "official_leaderboard_submission":False}
    score_path.write_text(json.dumps(score,indent=2,allow_nan=False)+"\n")
    return score


def gated_case_diagnostics(data_dir,entries,source_paths,coverage,split,output):
    """Post-prediction error diagnosis only; never changes a decision or a field."""
    archives={name:np.load(path,allow_pickle=False) for name,path in source_paths.items()}
    reports=[]
    try:
        for item,entry in zip(coverage["per_shot"],entries):
            if not item["gated_frames"]:
                continue
            i=item["validation_index"]
            idx=np.asarray(item["gate_frame_indices"],dtype=np.int64)
            with np.load(data_dir/entry["path"],allow_pickle=False) as shot:
                psi=shot["psi"].astype(np.float64)
                scalars=shot["scalars"].astype(np.float64)
            case={"shot_index":item["shot_index"],"shot_id":item["shot_id"],
                  "frames":item["frames"],"gated_frames":item["gated_frames"],"models":{}}
            for name,archive in archives.items():
                error=archive[f"shot_{i:04d}_psirz"].astype(np.float64)-psi
                scalar_pred=np.column_stack([archive[f"shot_{i:04d}_q95"],archive[f"shot_{i:04d}_betaN"]])
                scalar_error=scalar_pred-scalars
                case["models"][name]={"flux_rmse_all_frames":float(np.sqrt(np.mean(error**2))),
                    "flux_rmse_gated_frames":float(np.sqrt(np.mean(error[idx]**2))),
                    "q95_rmse_gated_frames":float(np.sqrt(np.nanmean(scalar_error[idx,0]**2))),
                    "betaN_rmse_gated_frames":float(np.sqrt(np.nanmean(scalar_error[idx,1]**2)))}
            reports.append(case)
    finally:
        for archive in archives.values():
            archive.close()
    (output/f"{split}_gated_cases.json").write_text(json.dumps({"role":"post-prediction validation error analysis only",
        "decisions_depend_on_targets":False,"cases":reports},indent=2,allow_nan=False)+"\n")
    return reports


def plot_and_report(summary,output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":11,"pdf.fonttype":42})
    labels={"full":"Full inputs","currents":"Currents only","gate":"Quality gate ≤10%","blend75":"75% full +25% currents"}
    methods=list(labels)
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout="constrained")
    for ax,split in zip(axes,SPLITS):
        result=summary["splits"][split]
        scores=[result["scores"][method]["S"] for method in methods]
        bars=ax.bar(range(4),scores,color=["#287D8E","#6F60A7","#12A89D","#E79938"])
        for bar,value in zip(bars,scores):
            ax.annotate(f"{value:.6f}",(bar.get_x()+bar.get_width()/2,value),xytext=(0,5),textcoords="offset points",ha="center",fontsize=10)
        ax.set_xticks(range(4),[labels[name] for name in methods],rotation=20,ha="right")
        ax.set_ylim(min(scores)-.01,1.002)
        ax.set_ylabel("Official local composite S")
        title="Fresh 128-shot validation" if split=="val" else "Legacy 32-shot validation"
        cov=result["coverage"]
        ax.set_title(f"{title}\nGate selects currents: {cov['gated_frames']}/{cov['frames']} frames")
        ax.grid(axis="y",color="#DDE3E8");ax.set_axisbelow(True)
        ax.spines[["top","right"]].set_visible(False)
    fig.suptitle("One predeclared diagnostic-quality gate and one fixed blend\nNo threshold search; input-only decisions; local scores do not establish leaderboard rank",fontsize=13,fontweight="bold")
    for ext in ("png","pdf"):
        fig.savefig(output/f"quality_gate_comparison.{ext}",dpi=200,bbox_inches="tight")
    plt.close(fig)
    lines=["# Input-quality gate experiment","",f"Created: {summary['created_utc']}","",
           "The fixed gate uses the currents-only expert when at most 10% of real edge Thomson channels have both positive, observed Te and ne. Otherwise it uses the full-input expert. Geometry/observation masks exclude padded channels. The rule uses no target values, shot identities, or prediction-based corrections. Threshold 0.1 and the alternative 75% full / 25% currents blend were fixed before scoring; no grid search was run.","",
           "| Fold | Method | S | Flux R² | Scalar R² | D_LCFS | Consistency |","|---|---|---:|---:|---:|---:|---:|"]
    for split in SPLITS:
        for method in methods:
            score=summary["splits"][split]["scores"][method]
            lines.append(f"| {split} | {labels[method]} | {score['S']:.6f} | {score['r2_psi']:.6f} | {score['r2_qb']:.6f} | {score['dlcfs']:.6f} | {score['consistency']:.6f} |")
    lines += ["","## Gate coverage",""]
    for split in SPLITS:
        c=summary["splits"][split]["coverage"]
        lines.append(f"- {split}: {c['gated_frames']} of {c['frames']} frames in {c['gated_shots']} of {c['shots']} shots select the currents-only expert.")
    if not summary["splits"]["val"]["coverage"]["gated_frames"]:
        lines += ["","**The gate never activates on the fresh fold.** Its fresh score therefore reproduces the full-input expert and provides no independent test of the low-quality regime. Legacy improvements in gated cases remain development evidence; the reserved audit was not read."]
    lines += ["","[Comparison plot](quality_gate_comparison.png) · [PDF](quality_gate_comparison.pdf)","",
              "`policy.json` records the rule and source hashes before scoring. `summary.json` retains exact score paths and coverage. `*_gated_cases.json` reports post-prediction errors on activated shots; those labels never alter the gate. Pure experts were rescored on the same frames with the unmodified official evaluator."]
    (output/"QUALITY_GATE.md").write_text("\n".join(lines)+"\n")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir",type=Path,default=ROOT/"data/advanced_v1")
    parser.add_argument("--full-dir",type=Path,default=ROOT/"results/optimization/pod_full_v1")
    parser.add_argument("--currents-dir",type=Path,default=ROOT/"results/optimization/pod_currents64_cpu")
    parser.add_argument("--output",type=Path,default=ROOT/"results/optimization/quality_gate_v1")
    parser.add_argument("--workers",type=int,default=1)
    args=parser.parse_args()
    if args.workers<1:parser.error("workers must be positive")
    for name in ("data_dir","full_dir","currents_dir","output"):
        setattr(args,name,getattr(args,name).resolve())
    args.output.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((args.data_dir/"manifest.json").read_text())
    splits=manifest.get("splits",manifest)
    source_paths={split:{"full":args.full_dir/f"{split}_predictions.npz",
                         "currents":args.currents_dir/f"{split}_predictions.npz"} for split in SPLITS}
    sources={split:{name:{"path":str(path),"sha256":sha256(path)} for name,path in experts.items()} for split,experts in source_paths.items()}
    policy={"created_before_scoring_utc":datetime.now(timezone.utc).isoformat(),
            "threshold":THRESHOLD,"fixed_blend_full_weight":BLEND_FULL_WEIGHT,
            "rule":"currents expert iff useful_edge_channel_count / real_edge_channel_count <=0.10; else full expert",
            "useful_channel":"Te>0 AND ne>0 AND Te observed AND ne observed",
            "real_channel":"geometry observed OR Te observed OR ne observed; excludes all-zero padding",
            "no_real_channels":"quality fraction0; choose currents expert",
            "gate_arguments":["allowed cached input features only"],
            "threshold_basis":"predeclared from previously documented zero-edge-Thomson development failure regime",
            "threshold_search":False,"target_adjustment":False,"shot_id_used_for_decision":False,
            "audit_labels_read":False,"source_manifest_sha256":sha256(args.data_dir/"manifest.json"),
            "script_sha256":sha256(Path(__file__)),"sources":sources}
    policy_path=args.output/"policy.json"
    policy_path.write_text(json.dumps(policy,indent=2)+"\n")
    policy_hash=sha256(policy_path)
    summary={"created_utc":datetime.now(timezone.utc).isoformat(),
             "evaluation_type":"predeclared_input_quality_gate_and_fixed_blend",
             "policy_sha256":policy_hash,"audit_labels_read":False,"leaderboard_submission":False,"splits":{}}
    # Materialize both folds using input-only decisions before evaluating any target.
    for split in SPLITS:
        pred,coverage=build_predictions(args.data_dir,splits[split],manifest["feature_names"],
            source_paths[split]["full"],source_paths[split]["currents"],args.output,split)
        source_paths[split].update(pred)
        summary["splits"][split]={"coverage":coverage,"scores":{},"score_paths":{}}
        print(f"{split}: gate selects currents on {coverage['gated_frames']}/{coverage['frames']} frames",flush=True)
    for split in SPLITS:
        for method,pred in source_paths[split].items():
            score_path=args.output/method/f"{split}_score.json"
            score=evaluate(args.data_dir,split,pred,score_path,args.workers,policy_hash,method)
            summary["splits"][split]["scores"][method]={k:v for k,v in score.items() if k not in ("evaluation","experiment")}
            summary["splits"][split]["score_paths"][method]=str(score_path)
            print(f"{split} {method}: S={score['S']:.6f}",flush=True)
        cases=gated_case_diagnostics(args.data_dir,splits[split],source_paths[split],summary["splits"][split]["coverage"],split,args.output)
        summary["splits"][split]["gated_case_count"]=len(cases)
    for split in SPLITS:
        for name in ("full","currents"):
            if sha256(source_paths[split][name])!=sources[split][name]["sha256"]:
                raise ValueError("An expert file changed during the experiment; discard this comparison and rerun")
    (args.output/"summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n")
    plot_and_report(summary,args.output)
    print(f"Completed predeclared gate comparison: {args.output}",flush=True)


if __name__=="__main__":
    main()
