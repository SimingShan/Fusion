#!/usr/bin/env python3
"""Bounded public-input MAST physics feasibility checks; not an official submission."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
from fusion.mast_physics import (
    MU0, FreeBoundaryGS, coil_response, read_mast_inputs, reconstruct_frame, ring_green, _surface,
)
from fusion.mast_structure import check_separatrix


def serializable(value):
    if isinstance(value, dict):
        return {k: serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def analytic_checks():
    # Independent Biot-Savart line quadrature for the vector potential.
    theta = (np.arange(32768)+0.5)*2*np.pi/32768
    R, Z, a, b = 0.75, 0.31, 1.2, -0.4
    distance = np.sqrt(R*R+a*a-2*R*a*np.cos(theta)+(Z-b)**2)
    direct = MU0/(4*np.pi)*R*a*np.mean(np.cos(theta)/distance)*2*np.pi
    green = float(ring_green(R,Z,a,b))
    relative_error = abs(green-direct)/abs(direct)
    reciprocal = float(ring_green(a,b,R,Z))
    assert relative_error < 1e-11
    assert abs(green-reciprocal) < 1e-20
    Rv, Zv = np.linspace(0.06,2,65),np.linspace(-2,2,65)
    solver = FreeBoundaryGS(Rv,Zv)
    j = np.exp(-((solver.RR-0.85)/0.22)**2-(solver.ZZ/0.36)**2)
    j.ravel()[solver.boundary] = 0
    j *= 6e5/(j.sum()*solver.dR*solver.dZ)
    psi = solver.solve(j)
    residual = solver.residual(psi,j)
    assert residual < 1e-10
    assert np.isfinite(psi).all()
    # Compare independently integrated full free-space kernel at a grid corner.
    rb,zb=Rv[-1],Zv[-1]
    manual=0.
    for u in (-1/np.sqrt(3),1/np.sqrt(3)):
        for v in (-1/np.sqrt(3),1/np.sqrt(3)):
            manual += np.sum(ring_green(rb,zb,solver.RR[1:-1,1:-1]+u*solver.dR/2,
                                        solver.ZZ[1:-1,1:-1]+v*solver.dZ/2)*j[1:-1,1:-1])
    manual *= solver.dR*solver.dZ/4
    assert abs(psi[-1,-1]-manual)<1e-10
    dummy_geometry={'coil_R':np.array([10.]),'coil_Z':np.array([10.]),
                    'coil_width':np.array([.01]),'coil_height':np.array([.01])}
    topology_checks={}
    for name,quartic,expected in [('diverted',1.,True),('limited',0.,False)]:
        field=-(solver.RR-1)**2-solver.ZZ**2+quartic*solver.ZZ**4
        surface=_surface(field,solver,np.ones(solver.shape,bool))
        assert surface is not None
        contour,_,edge,delta,_=surface
        structure=check_separatrix(field,Rv,Zv,contour,edge,delta,dummy_geometry)
        assert structure['separatrix_supported']==expected
        topology_checks[name]={'separatrix_supported':structure['separatrix_supported'],
                              'saddle_count':len(structure['saddles'])}
    return {"ring_Biot_Savart_relative_error":relative_error,
            "ring_reciprocity_absolute_error":abs(green-reciprocal),
            "synthetic_linear_GS_relative_residual":residual,
            "synthetic_boundary_integral_absolute_error":abs(psi[-1,-1]-manual),
            "synthetic_Ip_A":float(j.sum()*solver.dR*solver.dZ),
            "synthetic_topology_checks":topology_checks}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw",type=Path,default=ROOT/"data/raw")
    p.add_argument("--output",type=Path,default=ROOT/"results/challenge2/physics_prototype_v1")
    p.add_argument("--shots",type=int,default=4)
    p.add_argument("--frames",type=int,default=3)
    p.add_argument("--iterations",type=int,default=30)
    p.add_argument("--self-check-only",action="store_true")
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    started=time.time()
    checks=analytic_checks()
    (args.output/"analytic_checks.json").write_text(json.dumps(checks,indent=2)+"\n")
    print("Analytic checks",json.dumps(checks),flush=True)
    if args.self_check_only:return
    manifest=json.loads((args.raw/"download_manifest.json").read_text())
    files=sorted(x["path"] for x in manifest["files"] if x["path"].startswith("data/mast_public_test/") and x["path"].endswith(".parquet"))
    indices=np.unique(np.linspace(0,len(files)-1,args.shots).astype(int))
    records=[]
    snapshots={}
    for shot_index in indices:
        path=args.raw/files[shot_index]
        row=read_mast_inputs(path)
        response=coil_response(row)
        solver=FreeBoundaryGS(row["efit_grid_R"],row["efit_grid_Z"])
        frames=np.unique(np.linspace(0,len(row["efit_times"])-1,args.frames).astype(int))
        for frame in frames:
            record={"canonical_shot_index":int(shot_index),"frame_index":int(frame),"source":str(path),
                    "source_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
                    "geometry_sha256":response.geometry_sha256,"conductor_counts":response.conductor_counts}
            try:
                result=reconstruct_frame(row,int(frame),response,solver,iterations=args.iterations)
                record.update(result["diagnostics"])
                key=f"shot_{shot_index:04d}_frame_{frame:03d}"
                for name in ("psi","vacuum","initial_psi","jtor"):
                    snapshots[f"{key}_{name}"]=result[name]
                snapshots[f"{key}_R"]=solver.R
                snapshots[f"{key}_Z"]=solver.Z
            except ValueError as e:
                record.update({"status":"unsupported_input_or_closure","error":str(e),"converged":False})
            records.append(record)
            print(json.dumps({k:record.get(k) for k in ("canonical_shot_index","frame_index","status","linear_GS_relative_residual","nonlinear_profile_relative_residual","q95","betaN")}),flush=True)
    np.savez_compressed(args.output/"diagnostic_fields.npz",**snapshots)
    report={"purpose":"Input-only MAST physics feasibility; no local or official accuracy score",
            "label_role":"No MAST labels, no demo labels, no audit labels; public inputs only",
            "revision":manifest["revision"],"selected_shot_indices":indices.tolist(),
            "selection":"Deterministic uniform public-input shot/frame diagnostic sampling",
            "settings":{"iterations":args.iterations,"pressure_ion_to_electron_ratio":1.0,
                        "profile_exponent":1.0,"relaxation":0.2,"TF_turns":24},
            "analytic_checks":checks,"records":records,"elapsed_seconds":time.time()-started,
            "converged_frames":sum(bool(r.get("converged")) for r in records),
            "closed_surface_frames":sum(bool(r.get("closed_surface")) for r in records),
            "total_frames":len(records),"submission_ready":False}
    (args.output/"summary.json").write_text(json.dumps(serializable(report),indent=2,allow_nan=False)+"\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if snapshots:
        key=next(iter(snapshots)).rsplit("_",1)[0]
        R,Z=snapshots[key+"_R"],snapshots[key+"_Z"]
        fig,axes=plt.subplots(1,3,figsize=(12,4),constrained_layout=True)
        for ax,name,title in zip(axes,("vacuum","initial_psi","psi"),
                                  ("Analytic coil flux","Coils + assumed plasma profile","After GS closure iterations")):
            field=snapshots[key+"_"+name]
            cs=ax.contourf(R,Z,field,levels=30,cmap="RdBu_r")
            ax.contour(R,Z,field,levels=20,colors="k",linewidths=0.4)
            ax.set(xlabel="R [m]",ylabel="Z [m]",title=title)
            ax.set_aspect("equal")
            fig.colorbar(cs,ax=ax,label="ψ [Wb/rad]")
        fig.suptitle("MAST input-only feasibility prototype — accuracy unmeasured")
        fig.savefig(args.output/"prototype_flux.png",dpi=160)
        plt.close(fig)
    print("Completed",args.output,"converged",report["converged_frames"],"of",len(records),flush=True)


if __name__=="__main__":main()
