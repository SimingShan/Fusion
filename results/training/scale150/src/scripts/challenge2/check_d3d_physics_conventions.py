#!/usr/bin/env python3
"""DIII-D training-label diagnostic of Green-function units and gauge, never MAST fitting.

The plasma source is differentiated from TRAINING target flux: it is an oracle
diagnostic of units, not a deployable predictor or a validation score.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
import pyarrow.parquet as pq
from fusion.data import arrow_to_numpy, interpolate_signal
from fusion.mast_physics import MU0, FreeBoundaryGS, ring_green
from starter.data_fixes import fix_d3d_ip_times
from derive import _points_in_polygon
from lcfs import extract_lcfs


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--shots",type=int,default=8)
    p.add_argument("--output",type=Path,default=ROOT/"results/challenge2/d3d_physics_conventions_v1")
    p.add_argument("--coil-geometry", choices=("filament", "solenoid_rectangle"), default="filament")
    p.add_argument("--normalize-current-orientation", action="store_true")
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((ROOT/"data/advanced_v1/manifest.json").read_text())
    entries=manifest["train"]
    selected=[entries[i] for i in np.linspace(0,len(entries)-1,args.shots).astype(int)]
    records=[];fit=[];individual_fit=[];kernel_cache={};solenoid_design=[]
    for shotnum,entry in enumerate(selected):
        path=ROOT/"data/raw"/entry["shard"]
        names=pq.ParquetFile(path).schema_arrow.names
        # Only DIII-D training paths from the fixed manifest can enter this read.
        assert "diii_d_train/" in str(path)
        allowed={"source","efit_psirz","efit_times","efit_grid_R","efit_grid_Z",
                 "magnetics_time","magnetics_plasma_current","magnetics_plasma_current_times",
                 "magnetics_ECOILA","coil_input_column","coil_R","coil_Z","coil_width","coil_height"}
        allowed|={f"magnetics_F{n}{side}" for n in range(1,10) for side in "AB"}
        columns=sorted(allowed.intersection(names));table=pq.read_table(path,columns=columns)
        row={key:arrow_to_numpy(table[key][0]) for key in columns}
        solver=FreeBoundaryGS(row["efit_grid_R"],row["efit_grid_Z"])
        with np.load(ROOT/"starter/fusion_scoring/masks/d3d_envelope.npz") as data:
            mask=data["mask_coarse"].astype(bool)
        R,Z=solver.R,solver.Z
        kernels=[]
        for c,(key,a,b) in enumerate(zip(row["coil_input_column"],row["coil_R"],row["coil_Z"])):
            width,height=float(row["coil_width"][c]),float(row["coil_height"][c])
            cache_key=(R.tobytes(),Z.tobytes(),key,float(a),float(b),width,height,args.coil_geometry)
            if cache_key not in kernel_cache:
                if args.coil_geometry=="solenoid_rectangle" and key=="magnetics_ECOILA":
                    if float(a)-width/2<=0 or width<=0 or height<=0:
                        raise ValueError("Invalid solenoid rectangle")
                    xr,wr=np.polynomial.legendre.leggauss(8)
                    xz,wz=np.polynomial.legendre.leggauss(64)
                    response=np.zeros_like(solver.RR)
                    for ri,wi in zip(float(a)+xr*width/2,wr/2):
                        values=ring_green(solver.RR[None],solver.ZZ[None],ri,
                                          (float(b)+xz*height/2)[:,None,None])
                        response+=wi*np.einsum("n,nzr->zr",wz/2,values)
                    kernel_cache[cache_key]=response
                else:
                    kernel_cache[cache_key]=ring_green(solver.RR,solver.ZZ,float(a),float(b))
            kernels.append(kernel_cache[cache_key])
        dt=fix_d3d_ip_times(row)
        for frame in np.unique(np.linspace(0,len(row["efit_times"])-1,3).astype(int)):
            t=float(row["efit_times"][frame]);phi=-np.asarray(row["efit_psirz"][frame],float)
            C=extract_lcfs(phi,R,Z,"DIII-D",mask,mask.astype(float),axis_sign=1.)
            if C is None:continue
            inside=_points_in_polygon(solver.RR.ravel(),solver.ZZ.ravel(),C).reshape(phi.shape)
            # Native second-order Delta* with the same matrix as the independent solve.
            j=-(solver.matrix@phi.ravel()).reshape(phi.shape)/(MU0*solver.RR)
            j[~inside]=0.;j.ravel()[solver.boundary]=0.
            plasma=solver.solve(j)
            ip,_=interpolate_signal(dt,row["magnetics_plasma_current"],np.array([t]))
            orientation=float(np.sign(ip[0])) if args.normalize_current_orientation else 1.0
            if orientation==0:
                raise ValueError("Zero plasma current cannot define the physical orientation")
            pf=np.zeros_like(phi);solenoid=np.zeros_like(phi);components=[]
            solenoid_unit=None
            for key,kernel in zip(row["coil_input_column"],kernels):
                current,_=interpolate_signal(row["magnetics_time"],row[key],np.array([t]))
                component=kernel*float(current[0])*1000*orientation
                components.append(component)
                if key=="magnetics_ECOILA":
                    solenoid+=component
                    solenoid_unit=kernel
                else:pf+=component
            exterior=~mask
            target=(phi-plasma)[exterior]
            regressors=np.stack([pf[exterior],solenoid[exterior]],axis=1)
            centered=regressors-regressors.mean(0)
            fit.append((centered,target-target.mean(),regressors.mean(0),float(target.mean())))
            indiv=np.stack([c[exterior] for c in components],axis=1)
            individual_fit.append((indiv-indiv.mean(0),indiv.mean(0)))
            solenoid_design.append(solenoid_unit[exterior]-solenoid_unit[exterior].mean())
            records.append({"training_shot_index":entry["shot_index"],"frame_index":int(frame),
                            "source":str(path),"subfold":"coefficient_fit" if shotnum%2==0 else "coefficient_check",
                            "Ip_measured_A":float(ip[0]*1000),"Ip_from_target_Delta_star_A":float(j.sum()*solver.dR*solver.dZ),
                            "applied_current_orientation":orientation,
                            "Ip_ratio":float(j.sum()*solver.dR*solver.dZ/(ip[0]*1000*orientation)),
                            "phi_mean":float(phi.mean()),"plasma_green_mean":float(plasma.mean()),
                            "PF_green_mean":float(pf.mean()),"ECOILA_unit_green_mean":float(solenoid.mean())})
    fit_indices=[i for i,r in enumerate(records) if r["subfold"]=="coefficient_fit"]
    X=np.concatenate([fit[i][0] for i in fit_indices]);y=np.concatenate([fit[i][1] for i in fit_indices])
    scales=np.linalg.lstsq(X,y,rcond=None)[0]
    Xi=np.concatenate([individual_fit[i][0] for i in fit_indices])
    individual_scales=np.linalg.lstsq(Xi,y,rcond=1e-10)[0]
    for i,record in enumerate(records):
        Xc,yc,xmean,ymean=fit[i]
        residual=yc-Xc@scales
        record["exterior_centered_RMSE_Wb_per_rad"]=float(np.sqrt(np.mean(residual**2)))
        record["required_frame_gauge_offset"]=float(ymean-xmean@scales)
        ic,im=individual_fit[i]
        record["individual_coil_centered_RMSE"]=float(np.sqrt(np.mean((yc-ic@individual_scales)**2)))
        record["individual_coil_required_gauge_offset"]=float(ymean-im@individual_scales)
        # Isolate the omitted co-located ECOILB ambiguity without exporting a calibration.
        # Both PF scale and the total solenoid response are fit separately for each
        # training frame; this is a best-case diagnostic using target information.
        flexible=np.stack([Xc[:,0],solenoid_design[i]],axis=1)
        best=np.linalg.lstsq(flexible,yc,rcond=None)[0]
        record["per_frame_PF_and_solenoid_diagnostic_RMSE"]=float(np.sqrt(np.mean((yc-flexible@best)**2)))
        record["per_frame_PF_diagnostic_factor"]=float(best[0])
        record["per_frame_solenoid_diagnostic_ampere_turns"]=float(best[1])
    result={"purpose":"Training-target source diagnostic; not deployable prediction and not a validation score",
            "label_role":"DIII-D train labels only; no MAST/demo/test/audit labels",
            "coil_geometry_mode":args.coil_geometry,
            "normalize_current_orientation":args.normalize_current_orientation,
            "assumptions":["DIII-D phi=-stored_psi (published maximum-axis convention)",
                "Native-grid Delta* differentiates training target and clips source to its extracted training LCFS",
                "DIII-D PF coils treated as center filaments; six parallelogram cross-sections omitted",
                "ECOILA uses a center filament or an 8x64-point uniform rectangle quadrature as explicitly selected",
                "F currents multiplied by 1000 once (already kA-turn)",
                "ECOILA unresolved effective response estimated, not guessed from a turn count",
                "An independent per-frame spatially constant offset is removed only for this diagnostic fit"],
            "fitted_PF_global_response_factor":float(scales[0]),
            "fitted_ECOILA_effective_response_factor":float(scales[1]),
            "individual_coil_response_factors":dict(zip(row["coil_input_column"].tolist(),individual_scales.tolist())),
            "individual_response_design_condition":float(np.linalg.cond(Xi)),
            "fit_shots_even_diagnostic_order_check_shots_odd":True,
            "records":records,"MAST_gauge_inferred":False,
            "source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (args.output/"summary.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    (args.output/"source.py").write_bytes(Path(__file__).read_bytes())
    print(json.dumps({"PF_factor":float(scales[0]),"ECOILA_factor":float(scales[1]),
                      "median_Ip_ratio":float(np.median([r["Ip_ratio"] for r in records])),
                      "median_exterior_centered_RMSE":float(np.median([r["exterior_centered_RMSE_Wb_per_rad"] for r in records])),
                      "gauge_offset_range":[min(r["required_frame_gauge_offset"] for r in records),max(r["required_frame_gauge_offset"] for r in records)]}),flush=True)


if __name__=="__main__":main()
