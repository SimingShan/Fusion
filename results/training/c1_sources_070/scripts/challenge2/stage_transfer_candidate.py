#!/usr/bin/env python3
"""Stage a reviewed complete transfer export with the best verified DIII-D payload."""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import copy
import json
import re
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.periodic_submission import digest,candidate_evidence,preserves_best_diii_d
from scripts.codabench_submit import write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--export",type=Path,required=True)
    p.add_argument("--candidate",required=True)
    p.add_argument("--review",type=Path,required=True)
    a=p.parse_args()
    if not re.fullmatch(r"[0-9]{8}-[0-9]{3}",a.candidate):p.error("Use a numeric candidate identifier")
    root=ROOT/"results/candidates";destination=root/a.candidate
    if destination.exists():raise ValueError("Use a fresh candidate directory")
    review=json.loads(a.review.read_text())
    if review.get("decision")!="accept" or review.get("mast_labels_read") is not False or review.get("audit_labels_read") is not False:
        raise ValueError("An explicit independent development review is required")
    for key in ("development_physics_report","mast_coverage_report"):
        evidence=review[key]
        if digest(ROOT/evidence["path"])!=evidence["sha256"]:raise ValueError("Reviewed evidence changed")
    if not review.get("rationale") or review.get("mast_accuracy_established") is not False:
        raise ValueError("State the rationale and distinguish unverified candidate accuracy")
    exported=json.loads((a.export/"mast_export.json").read_text())
    if exported["status"]!="complete" or exported["scope"]!="full" or exported["shots"]!=1206 or exported["frames"]!=69075:
        raise ValueError("Only a complete native MAST export can be staged")
    if exported["scalar_valid"]!=69075 or exported["runtime_limited_frames"]:
        raise ValueError("Incomplete scalars or runtime-dependent predictions")
    plan=json.loads((a.export/"export_plan.json").read_text())
    if digest(a.export/"export_plan.json")!=exported["plan_sha256"] or digest(a.export/"checkpoints/prior.joblib")!=exported["prior_sha256"]:
        raise ValueError("Frozen MAST inference changed")
    for name,expected in plan["source_sha256"].items():
        if digest(a.export/"source_snapshot"/name)!=expected:raise ValueError("Frozen MAST source changed")
    if (a.export/"reuse_proof.json").exists():
        reuse=json.loads((a.export/"reuse_review.json").read_text())
        if (reuse.get("passed") is not True or reuse.get("new_plan_sha256")!=exported["plan_sha256"]
                or reuse.get("frames")!=69075 or reuse.get("shots")!=1206
                or digest(a.export/"reuse_proof.json")!=exported.get("reuse_proof_sha256")
                or digest(a.export/"reuse_review.json")!=exported.get("reuse_review_sha256")):
            raise ValueError("Reused predictions lack a complete checked provenance record")
    mast=a.export/"mast_public_test.npz"
    if digest(mast)!=exported["prediction_sha256"]:raise ValueError("MAST payload changed")
    state=json.loads((root/"submission_state.json").read_text())
    best=state["historical_best"]["d3d_S"]["submission_id"]
    names=[name for name,entry in state["entries"].items() if entry.get("verified_success") and entry.get("submission_id")==best]
    if len(names)!=1:raise ValueError("A unique verified DIII-D source is required")
    baseline=root/names[0]
    original=json.loads((baseline/"decision.json").read_text())
    d3export=json.loads((baseline/"export.json").read_text())
    if not preserves_best_diii_d(root,state,d3export["prediction_sha256"]):raise ValueError("DIII-D reference changed")
    d3source=Path(d3export["prediction"])
    if digest(d3source)!=d3export["prediction_sha256"]:raise ValueError("Verified DIII-D payload changed")
    (destination/"checkpoints").mkdir(parents=True)
    records=[]
    for old,name in zip(original["checkpoints"],("a.pt","b.pt")):
        source=ROOT/old["frozen_file"]
        if digest(source)!=old["sha256"]:raise ValueError("Verified DIII-D checkpoint changed")
        target=destination/"checkpoints"/name;shutil.copy2(source,target)
        records.append({"frozen_file":str(target),"sha256":digest(target)})
    shutil.copy2(d3source,destination/"diii_d_public_test.npz")
    shutil.copy2(mast,destination/"mast_public_test.npz")
    d3export.update(primary=records[0]["frozen_file"],secondary=records[1]["frozen_file"],
                    prediction=str(destination/"diii_d_public_test.npz"))
    write_json(destination/"export.json",d3export)
    decision=copy.deepcopy(original)
    decision.update(candidate=a.candidate,public_name="Submission "+a.candidate,
                    frozen_utc=datetime.now(timezone.utc).isoformat(),checkpoints=records,
                    rationale="Add a reviewed first transfer baseline while preserving the best verified DIII-D payload")
    decision["validation"]["baseline_native"]=copy.deepcopy(decision["validation"]["native"])
    decision["validation"]["baseline_legacy"]=copy.deepcopy(decision["validation"]["legacy"])
    write_json(destination/"decision.json",decision)
    parity=json.loads((baseline/"export_parity.json").read_text())
    parity.update(unchanged_verified_diii_d_payload=True,baseline_submission_id=best)
    write_json(destination/"export_parity.json",parity)
    exported.update(prediction=str(destination/"mast_public_test.npz"),frozen_export_source=str(a.export.resolve()))
    write_json(destination/"mast_export.json",exported)
    for name in ("reuse_proof.json","reuse_review.json"):
        if (a.export/name).exists():shutil.copy2(a.export/name,destination/name)
    review.update(baseline_diii_d_prediction_sha256=d3export["prediction_sha256"],baseline_submission_id=best,
                  full_export_source=str(a.export.resolve()),full_export_plan_sha256=exported["plan_sha256"])
    write_json(destination/"transfer_review.json",review)
    subprocess.run([sys.executable,str(ROOT/"scripts/package_dual_submission.py"),"--diii-d",
                    str(destination/"diii_d_public_test.npz"),"--mast",str(destination/"mast_public_test.npz"),
                    "--output",str(destination/"submission.zip")],check=True)
    evidence=candidate_evidence(destination)
    if evidence is None or not evidence["dual"]:raise ValueError("The complete candidate did not pass queue admission")
    print(json.dumps({"candidate":a.candidate,"ready_for_periodic_submission":True,"baseline_submission_id":best}),flush=True)


if __name__=="__main__":
    main()
