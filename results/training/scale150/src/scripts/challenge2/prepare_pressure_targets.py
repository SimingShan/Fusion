#!/usr/bin/env python3
"""Fit permitted DIII-D closure targets consistently with measured pressure."""
from pathlib import Path
import argparse
import copy
import hashlib
import json
import sys

import numpy as np
from scipy.optimize import nnls

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from fusion.mast_profiles import read_row, FreeBoundaryGS, MU0
from fusion.mast_physics import _surface
from fusion.measured_pressure import measurements, PressureConstraint
from fusion.smooth_profiles import basis, integral, current
from scripts.codabench_submit import write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--pressure-factor",type=float,default=2.)
    a=p.parse_args()
    if (a.output/"targets.npz").exists():
        raise ValueError("Use a fresh target directory")
    a.output.mkdir(parents=True,exist_ok=True)
    with np.load(a.source/"targets.npz",allow_pickle=False) as z:
        d=dict(z)
    metadata=copy.deepcopy(json.loads((a.source/"preparation.json").read_text()))
    if metadata.get("closure_basis")!="edge_vanishing":
        raise ValueError("Matching continuous-edge targets are required")
    solver=FreeBoundaryGS(d["R"],d["Z"])
    previous_path,row=None,None
    for i,record in enumerate(metadata["records"]):
        if not d["fit_valid"][i]:
            continue
        if record["fold"] not in {"train","val","legacy_val"}:
            raise ValueError("Only predeclared training and development partitions are permitted")
        path=ROOT/record["source"]
        if "diii_d_train" not in str(path):
            raise ValueError("Only released DIII-D training paths may supply supervision")
        if path!=previous_path:
            row=read_row(path,labels=False)
            previous_path=path
        phi=d["phi"][i].astype(float)
        ip,L=float(d["Ip_A"][i]),float(d["L"][i])
        previous_diagnostics={k:record.get(k) for k in ("source_relative_RMSE","pressure_axis_Pa","paxis_star")}
        record["unconstrained_pressure_target_diagnostics"]=previous_diagnostics
        try:
            surface=_surface(phi,solver,d["envelope"])
            if surface is None:
                raise ValueError("Training target has no closed surface")
            contour,_,edge,delta,inside=surface
            observed=measurements(row,record["time_ms"])
            prior=float(d["paxis_star"][i]*MU0*ip**2/L**2)
            constraint=PressureConstraint(observed,prior,d["A_shape"][i],a.pressure_factor)
            paxis,ashape,pressure_details=constraint.fit(phi,solver,d["envelope"],surface)
            if not pressure_details["pressure_fit_used"]:
                raise ValueError("Insufficient measured pressure for this supervised target")
            s=np.clip((phi-edge)/delta,0,1)
            B=basis(s)
            pprime=paxis/delta*(B@ashape)/(integral(1.)@ashape)
            pressure_current=(solver.RR-float(contour[:,0].min())**2/solver.RR)*pprime*inside
            if pressure_current.sum()*solver.dR*solver.dZ>=ip:
                raise ValueError("Measured pressure exceeds total-current constraint")
            target=-(solver.matrix@phi.ravel()).reshape(solver.shape)/(MU0*solver.RR)
            target*=inside
            target.ravel()[solver.boundary]=0
            total=target.sum()*solver.dR*solver.dZ
            if total<=0:
                raise ValueError("Nonpositive target current")
            # Match the actual current constraint. The unnormalized discrepancy
            # remains recorded and the original quality filter still applies.
            target*=ip/total
            selected=inside&(s>.025)&(s<.975)
            bins=np.minimum((s[selected]*12).astype(int),11)
            weights=1/np.sqrt(np.bincount(bins,minlength=12)[bins])
            design=B[selected]
            residual=((target-pressure_current)*solver.RR)[selected]
            X,y=design*weights[:,None],residual*weights
            scale=np.maximum(np.linalg.norm(X,axis=0),1e-8)
            cshape=nnls(np.vstack((X/scale,np.sqrt(.001)*np.eye(4))),
                         np.r_[y,np.zeros(4)],maxiter=1000)[0]/scale
            if cshape.sum()<=0:
                raise ValueError("No positive residual current profile")
            ashape/=ashape.sum()
            cshape/=cshape.sum()
            predicted=current(phi,solver,d["envelope"],ip,paxis,ashape,cshape)
            d["A_shape"][i],d["C_shape"][i]=ashape,cshape
            d["paxis_star"][i]=paxis/(MU0*ip**2/L**2)
            error=float(np.linalg.norm((predicted-target)[selected])/max(np.linalg.norm(target[selected]),1e-12))
            record.update(source_relative_RMSE=error,pressure_axis_Pa=float(paxis),
                          paxis_star=float(d["paxis_star"][i]),pressure_constraint=pressure_details,
                          target_source_normalized_to_observed_Ip=True,
                          fit_Ip_relative_error=float(abs(predicted.sum()*solver.dR*solver.dZ/ip-1)))
        except (ValueError,RuntimeError,np.linalg.LinAlgError) as exc:
            d["fit_valid"][i]=False
            record.update(fit_valid=False,pressure_target_failure=str(exc))
        if (i+1)%128==0:
            print(json.dumps({"frames_done":i+1}),flush=True)
    sources=[Path(__file__),ROOT/"fusion/measured_pressure.py",ROOT/"fusion/mast_profiles.py",
             ROOT/"fusion/mast_physics.py",ROOT/"fusion/smooth_profiles.py"]
    hashes={}
    for path in sources:
        relative=path.resolve().relative_to(ROOT.resolve())
        destination=a.output/"source_snapshot"/relative
        destination.parent.mkdir(parents=True,exist_ok=True)
        contents=path.read_bytes();destination.write_bytes(contents)
        hashes[str(relative)]=hashlib.sha256(contents).hexdigest()
    metadata.update(parent_preparation_sha256=hashlib.sha256((a.source/"preparation.json").read_bytes()).hexdigest(),
                    pressure_constraint_factor=a.pressure_factor,source_sha256=hashes,
                    inference_labels_used=False,audit_labels_read=False,mast_labels_read=False,
                    closure_target_note="Fit the residual current after the same input-pressure constraint used during inference")
    np.savez_compressed(a.output/"targets.npz",**d)
    write_json(a.output/"preparation.json",metadata)
    print(json.dumps({"frames":len(d["fit_valid"]),"fit_valid":int(d["fit_valid"].sum()),
                      "input_features_unchanged":True}),flush=True)


if __name__=="__main__":
    main()

