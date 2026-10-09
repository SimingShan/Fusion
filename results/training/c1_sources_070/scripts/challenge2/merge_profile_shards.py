#!/usr/bin/env python3
"""Merge complete native evaluation shards without changing any prediction."""
from pathlib import Path
import argparse
import hashlib
import json
import sys

import numpy as np
import pyarrow.parquet as pq

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.codabench_submit import write_json


def digest(path):
    with Path(path).open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--parts",type=int,required=True)
    a=p.parse_args()
    if (a.output/"summary.json").exists():raise ValueError("Use a fresh merged evaluation directory")
    parents=[a.source/f"part_{i}" for i in range(a.parts)]
    values=[json.loads((path/"summary.json").read_text()) for path in parents]
    baseline=values[0]
    settings=baseline["settings"]
    invariant={k:v for k,v in settings.items() if k not in ("output","shard_index")}
    if settings["frames"]!=0 or settings["shard_count"]!=a.parts:
        raise ValueError("Complete native shards are required")
    for i,value in enumerate(values):
        if value["settings"]["shard_index"]!=i or value["source_sha256"]!=baseline["source_sha256"] or value["prior_sha256"]!=baseline["prior_sha256"]:
            raise ValueError("Shard provenance differs")
        if {k:v for k,v in value["settings"].items() if k not in ("output","shard_index")}!=invariant:
            raise ValueError("Shard inference settings differ")
        if value["audit_labels_read"] or value["mast_labels_read"]:
            raise ValueError("Forbidden target provenance")
    data=Path(settings.get("data_root",ROOT / "data"))
    manifest=json.loads((data/"advanced_v1/manifest.json").read_text())
    inventory=json.loads((data/"raw/download_manifest.json").read_text())
    selected=[]
    if "DIII-D" in settings["machines"]:
        for index in np.unique(np.linspace(0,len(manifest["val"])-1,settings["shots"]).astype(int)):
            path=data/"raw"/manifest["val"][index]["shard"];selected.append(("DIII-D",path))
    if "MAST" in settings["machines"]:
        pool=sorted(x["path"] for x in inventory["files"] if x["path"].startswith("data/mast_public_test/") and x["path"].endswith(".parquet"))
        for index in np.unique(np.linspace(0,len(pool)-1,settings["shots"]).astype(int)):
            selected.append(("MAST",data/"raw"/pool[index]))
    a.output.mkdir(parents=True,exist_ok=True)
    records=[];prediction_hashes={}
    for ordinal,(machine,path) in enumerate(selected):
        part=ordinal%a.parts
        group=[x for x in values[part]["records"] if x["machine"]==machine and x["shot"]==path.stem]
        group.sort(key=lambda x:x["frame"])
        count=len(pq.read_table(path,columns=["efit_times"]).to_pylist()[0]["efit_times"])
        if [x["frame"] for x in group]!=list(range(count)):
            raise ValueError("Native evaluation is missing or duplicates a frame")
        for record in group:
            name=f"{machine}_{path.stem}_{record['frame']:04d}.npz"
            source=parents[part]/name
            with np.load(source,allow_pickle=False) as z:
                if z["psi"].shape!=(65,65) or not np.isfinite(z["psi"]).all():
                    raise ValueError("Invalid native prediction")
            destination=a.output/name
            if destination.exists():raise ValueError("Merged output already contains predictions")
            destination.symlink_to(source.resolve())
            prediction_hashes[name]=digest(source)
        records.extend(group)
    if sum(len(x["records"]) for x in values)!=len(records):
        raise ValueError("Unexpected predictions outside the declared query set")
    report={**baseline,"records":records,"frames":len(records),"converged":sum(x.get("converged",False) for x in records),
            "settings":{**settings,"output":str(a.output),"shard_index":"merged"},
            "parents_sha256":{str(path/"summary.json"):digest(path/"summary.json") for path in parents},
            "prediction_sha256":prediction_hashes,"complete_native_queries":True,"eligible_for_submission":False}
    write_json(a.output/"summary.json",report)
    (a.output/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    print(json.dumps({"shots":len(selected),"frames":len(records),"converged":report["converged"],"complete_native_queries":True}),flush=True)


if __name__=="__main__":
    main()
