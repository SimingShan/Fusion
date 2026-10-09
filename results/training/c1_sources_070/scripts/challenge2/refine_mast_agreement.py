#!/usr/bin/env python3
"""Tighten closure residual on discrepant cold/continued states at fixed inputs."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
from fusion.mast_physics import read_mast_inputs,coil_response,FreeBoundaryGS,reconstruct_frame
from scripts.challenge2.run_mast_prototype import serializable


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=ROOT/'results/challenge2/solver_comparison_v1')
    p.add_argument('--output',type=Path,default=ROOT/'results/challenge2/solver_tolerance_v1')
    p.add_argument('--frames',nargs='+',type=int,default=[22,24])
    p.add_argument('--tolerance',type=float,default=1e-4)
    p.add_argument('--max-evaluations',type=int,default=2000)
    p.add_argument('--max-seconds',type=float,default=20)
    p.add_argument('--methods',nargs='+',default=['anderson','krylov'])
    p.add_argument('--initialization-groups',nargs='+',default=['segment_cold','segment_continued'])
    p.add_argument('--sequential',action='store_true',help='Continue from the previous strictly converged native frame')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    assert not (args.output/'summary.json').exists()
    meta=json.loads((args.source/'summary.json').read_text())
    source=np.load(args.source/'fields.npz')
    original={f"{r['group']}_{r['method']}_shot_{r['canonical_shot_index']:04d}_frame_{r['frame_index']:03d}":r
              for r in meta['records']}
    rawpath=ROOT/'data/raw'/next(r['source'] for r in meta['records'] if r['canonical_shot_index']==0)
    row=read_mast_inputs(rawpath);response=coil_response(row)
    solver=FreeBoundaryGS(row['efit_grid_R'],row['efit_grid_Z'])
    records=[];fields={};previous={}
    for frame in args.frames:
        for method in args.methods:
            for group in args.initialization_groups:
                key=f'{group}_{method}_shot_0000_frame_{frame:03d}'
                assert original[key]['source_sha256']==hashlib.sha256(rawpath.read_bytes()).hexdigest()
                prior=previous.get((method,group)) if args.sequential else None
                initial=prior[1] if prior is not None else source[key+'_jtor']
                result=reconstruct_frame(row,frame,response,solver,solver_method=method,
                    initial_current=initial,iterations=400,max_evaluations=args.max_evaluations,
                    max_seconds=args.max_seconds,solver_tolerance=args.tolerance)
                record={'frame_index':frame,'method':method,'initialization_source':key,
                        'previous_residual':original[key]['nonlinear_profile_relative_residual'],
                        'previous_q95':original[key]['q95'],**result['diagnostics']}
                record['continued_from_strict_frame']=prior[0] if prior is not None else None
                if result['diagnostics']['converged']:
                    previous[(method,group)]=(frame,result['jtor'])
                records.append(record)
                for name in ('psi','jtor'):fields[key+'_'+name]=result[name]
                (args.output/'records.partial.json').write_text(json.dumps(serializable(records),indent=2,allow_nan=False)+'\n')
                print(json.dumps({k:record[k] for k in ['frame_index','method','initialization_source','status',
                    'function_evaluations','nonlinear_profile_relative_residual','q95','previous_q95','structurally_admissible_equilibrium']}),flush=True)
    np.savez_compressed(args.output/'fields.npz',**fields)
    output={'purpose':'Tolerance-refinement diagnosis at identical public inputs; no MAST accuracy claim',
        'label_role':'No MAST/demo/hidden/audit labels','source':str(args.source),
        'source_sha256':hashlib.sha256((args.source/'summary.json').read_bytes()).hexdigest(),
        'selection':'User-specified native frames; initial source states come only from the documented public-input numerical solver comparison',
        'settings':vars(args)|{'source':str(args.source),'output':str(args.output)},'records':records,
        'code_sha256':{str(x.relative_to(ROOT)):hashlib.sha256(x.read_bytes()).hexdigest()
                       for x in [Path(__file__),ROOT/'fusion/mast_physics.py',ROOT/'fusion/mast_structure.py']}}
    (args.output/'summary.json').write_text(json.dumps(serializable(output),indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
