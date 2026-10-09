#!/usr/bin/env python3
"""Bounded full-native temporal-output comparison, with fixed settings."""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import sys
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.optimization.quality_gate import write_array
from scripts.codabench_submit import write_json


def smooth_field(values, times, weight):
    """Five-point quadratic fit in physical time; leave edges and gaps intact."""
    values, times = np.asarray(values), np.asarray(times, float)
    if values.shape[0] != len(times) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("Strictly increasing finite native timestamps required")
    result = values.copy()
    if len(times) < 5:
        return result
    spacing = np.diff(times)
    typical = float(np.median(spacing))
    for center in range(2, len(times)-2):
        ts = times[center-2:center+3]
        if np.max(np.diff(ts)) > 1.5*typical:
            continue
        relative = (ts-times[center])/typical
        design = np.column_stack((np.ones(5), relative, relative**2))
        weights = np.linalg.pinv(design)[0]
        fitted = np.tensordot(weights, values[center-2:center+3], axes=(0,0))
        result[center] = (1-weight)*values[center] + weight*fitted
    return result


def check():
    # Independent analytic quadratic, irregular native sampling, plus a gap.
    time = np.arange(21, dtype=float)+.07*np.sin(np.arange(21))
    amplitude = 2+3*time-.4*time**2
    values = amplitude[:,None,None]*np.arange(1,10).reshape(1,3,3)
    np.testing.assert_allclose(smooth_field(values,time,.5), values, atol=1e-10, rtol=1e-12)
    shifted = time.copy(); shifted[10:] += 100
    actual = smooth_field(values, shifted, .5)
    np.testing.assert_array_equal(actual[8:12], values[8:12])
    np.testing.assert_array_equal(actual[[0,1,-2,-1]], values[[0,1,-2,-1]])
    return {"passed":True,"irregular_quadratic_preserved":True,"gaps_and_edges_unchanged":True}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--workers", type=int, default=2)
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    if (a.output/"comparison.json").exists():
        raise ValueError("Use a fresh comparison directory")
    source=ROOT/"results/optimization/native_history128_li_tuned_v9/gated_primary50/val_predictions.npz"
    data=ROOT/"data/advanced_val_native_v3"
    manifest=json.loads((data/"manifest.json").read_text())
    settings={"weights":[.25,.5],"window_frames":5,"polynomial_degree":2,"maximum_gap_multiple":1.5,
              "changed_output":"flux only","selection":"Full-native fresh development only; no audit",
              "created_utc":datetime.now(timezone.utc).isoformat(),"check":check(),
              "source_sha256":hashlib.sha256(source.read_bytes()).hexdigest()}
    write_json(a.output/"settings.json", settings)
    (a.output/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    paths={weight:a.output/f"trial_{round(100*weight):02d}.npz" for weight in settings["weights"]}
    writers={weight:zipfile.ZipFile(path,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) for weight,path in paths.items()}
    try:
        with np.load(source,allow_pickle=False) as baseline:
            for i,entry in enumerate(manifest["val"]):
                with np.load(data/entry["path"],allow_pickle=False) as shot:
                    times,indices=shot["times"],shot["frame_indices"]
                if not np.array_equal(indices,np.arange(len(indices))):
                    raise ValueError("Complete native ordering required")
                field=baseline[f"shot_{i:04d}_psirz"]
                for weight,z in writers.items():
                    write_array(z,f"shot_{i:04d}_psirz",smooth_field(field,times,weight))
                    for name in ("q95","betaN"):
                        write_array(z,f"shot_{i:04d}_{name}",baseline[f"shot_{i:04d}_{name}"])
                if (i+1)%32==0:
                    print(json.dumps({"shots_done":i+1}),flush=True)
    finally:
        for z in writers.values():
            z.close()
    scores={}
    for weight,path in paths.items():
        result=path.with_suffix(".json")
        with path.with_suffix(".log").open("w") as out:
            subprocess.run([sys.executable,str(ROOT/"scripts/evaluate.py"),"--data-dir",str(data),
                            "--split","val","--pred",str(path),"--output",str(result),"--workers",str(a.workers)],
                           stdout=out,stderr=subprocess.STDOUT,check=True)
        d=json.loads(result.read_text())
        scores[str(weight)]={key:d[key] for key in ("S","r2_psi","r2_qb","dlcfs","consistency","lcfs_fail_frac")}
        print(json.dumps({"weight":weight,**scores[str(weight)]}),flush=True)
    write_json(a.output/"comparison.json",{"scores":scores,"baseline_native_S":.995516,
               "audit_labels_read":False,"test_labels_read":False,"eligible_for_submission":False,
               "reason":"Diagnostic only; complete legacy and frozen export parity required before selection"})


if __name__=="__main__":
    main()

