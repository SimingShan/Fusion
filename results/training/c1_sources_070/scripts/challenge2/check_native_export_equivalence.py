#!/usr/bin/env python3
"""Verify frozen native exports before starting the full-machine run."""
from pathlib import Path
import argparse
import json
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.codabench_submit import write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--original",type=Path,required=True)
    p.add_argument("--updated",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    plans=[json.loads((path/"export_plan.json").read_text()) for path in (a.original,a.updated)]
    reports=[json.loads((path/"mast_export.json").read_text()) for path in (a.original,a.updated)]
    for key in ("scope","shots","prior_sha256","inventory_sha256","max_evaluations","max_seconds","pressure_factor","allow_approximate_seed","packages"):
        if plans[0][key]!=plans[1][key]:raise ValueError("Export contracts differ")
    if any(x["status"]!="smoke_complete" or x["runtime_limited_frames"]!=0 for x in reports):
        raise ValueError("Complete deterministic smoke exports required")
    errors={"psirz":0.,"q95":0.,"betaN":0.}
    with np.load(a.original/"mast_public_test.npz",allow_pickle=False) as old, np.load(a.updated/"mast_public_test.npz",allow_pickle=False) as new:
        if set(old.files)!=set(new.files):raise ValueError("Export keys differ")
        for key in old.files:
            field=key.rsplit("_",1)[-1]
            before,after=old[key],new[key]
            np.testing.assert_allclose(after,before,rtol=1e-5,atol=2e-6 if field=="psirz" else 1e-5)
            errors[field]=max(errors[field],float(np.max(np.abs(after-before))))
    write_json(a.output,{"passed":True,"shots":reports[0]["shots"],"frames":reports[0]["frames"],
                        "max_absolute_errors":errors,"targets_read":False,
                        "original_prediction_sha256":reports[0]["prediction_sha256"],
                        "updated_prediction_sha256":reports[1]["prediction_sha256"]})
    print(json.dumps({"passed":True,"frames":reports[0]["frames"],"max_absolute_errors":errors}),flush=True)


if __name__=="__main__":
    main()

