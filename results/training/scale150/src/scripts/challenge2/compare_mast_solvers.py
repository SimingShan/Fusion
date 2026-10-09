#!/usr/bin/env python3
"""Prespecified label-free fixed-coil solver comparison and native continuation."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
from fusion.mast_physics import read_mast_inputs,coil_response,FreeBoundaryGS,reconstruct_frame
from scripts.challenge2.run_mast_prototype import analytic_checks,serializable


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'results/challenge2/solver_comparison_v1')
    parser.add_argument('--methods',nargs='+',default=['picard','anderson','krylov'])
    parser.add_argument('--max-evaluations',type=int,default=400)
    parser.add_argument('--max-seconds',type=float,default=12.)
    parser.add_argument('--iterations',type=int,default=200)
    parser.add_argument('--segment-start',type=int,default=18)
    parser.add_argument('--segment-frames',type=int,default=8)
    parser.add_argument('--pilot',action='store_true')
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'summary.json').exists():raise ValueError('Use a new output directory')
    started=time.monotonic()
    checks=analytic_checks()
    manifest=json.loads((ROOT/'data/raw/download_manifest.json').read_text())
    files=sorted(x['path'] for x in manifest['files'] if x['path'].startswith('data/mast_public_test/')
                 and x['path'].endswith('.parquet'))
    selected=[0] if args.pilot else [0,401,803,1205]
    records=[];fields={};cache={}
    def shot(index):
        if index not in cache:
            path=ROOT/'data/raw'/files[index]
            row=read_mast_inputs(path)
            response=coil_response(row)
            solver=FreeBoundaryGS(row['efit_grid_R'],row['efit_grid_Z'])
            cache[index]=(row,response,solver,hashlib.sha256(path.read_bytes()).hexdigest())
        return cache[index]
    def run(index,frame,method,group,initial=None,previous=None):
        row,response,solver,sha=shot(index)
        record={'canonical_shot_index':index,'frame_index':frame,'method':method,'group':group,
                'source':files[index],'source_sha256':sha,'geometry_sha256':response.geometry_sha256,
                'continued_from_frame':previous}
        try:
            result=reconstruct_frame(row,frame,response,solver,solver_method=method,initial_current=initial,
                iterations=args.iterations,max_evaluations=args.max_evaluations,max_seconds=args.max_seconds)
            record.update(result['diagnostics'])
            key=f'{group}_{method}_shot_{index:04d}_frame_{frame:03d}'
            for name in ('psi','jtor','vacuum'):fields[key+'_'+name]=result[name]
            fields[key+'_R']=solver.R;fields[key+'_Z']=solver.Z
        except ValueError as e:
            result=None
            record.update({'status':'unsupported_input:'+str(e),'converged':False,'scalar_valid':False,
                           'structurally_admissible_equilibrium':False,'closed_surface':False})
        records.append(record)
        (args.output/'records.partial.json').write_text(json.dumps(serializable(records),indent=2,allow_nan=False)+'\n')
        print(json.dumps({k:record.get(k) for k in ('group','canonical_shot_index','frame_index','method','status',
              'function_evaluations','nonlinear_profile_relative_residual','diverted_topology_verified',
              'structurally_admissible_equilibrium','q95','betaN','TS_pressure_relative_RMSE')}),flush=True)
        return result
    for index in selected:
        row,*_=shot(index)
        frames=[len(row['efit_times'])//2] if args.pilot else np.unique(np.linspace(0,len(row['efit_times'])-1,3).astype(int)).tolist()
        for frame in frames:
            for method in args.methods:run(index,frame,method,'same_frames')
    if args.segment_frames and not args.pilot:
        row,*_=shot(0)
        indices=list(range(args.segment_start,min(args.segment_start+args.segment_frames,len(row['efit_times']))))
        for method in [x for x in args.methods if x!='picard']:
            previous_current=None;previous_frame=None
            for frame in indices:
                run(0,frame,method,'segment_cold')
                result=run(0,frame,method,'segment_continued',previous_current,previous_frame)
                if result is not None and result['diagnostics']['converged']:
                    previous_current=result['jtor'].astype(float);previous_frame=frame
    aggregates=[]
    for group in sorted(set(r['group'] for r in records)):
        for method in args.methods:
            part=[r for r in records if r['group']==group and r['method']==method]
            if not part:continue
            aggregates.append({'group':group,'method':method,'frames':len(part),
                'converged':sum(r['converged'] for r in part),
                'closed_surfaces':sum(r['closed_surface'] for r in part),
                'scalar_valid':sum(r['scalar_valid'] for r in part),
                'structurally_admissible':sum(r['structurally_admissible_equilibrium'] for r in part),
                'total_function_evaluations':sum(r.get('function_evaluations',0) for r in part),
                'total_solver_seconds':sum(r.get('solver_elapsed_seconds',0) for r in part)})
    output={'purpose':'Numerical fixed-coil solver and physical-structure comparison, not a MAST accuracy score',
        'label_role':'Public MAST inputs only; no MAST/demo/hidden/audit labels',
        'dataset_revision':manifest['revision'],'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [Path(__file__),ROOT/'fusion/mast_physics.py',ROOT/'fusion/mast_structure.py']},
        'selection':'Prespecified four uniform shots and three uniform native frames; native segment shot0 frames18..25',
        'settings':vars(args)|{'output':str(args.output)},'analytic_checks':checks,'aggregates':aggregates,
        'records':records,'elapsed_seconds':time.monotonic()-started,'submission_ready':False}
    (args.output/'summary.partial.json').write_text(json.dumps(serializable(output),indent=2,allow_nan=False)+'\n')
    os.replace(args.output/'summary.partial.json',args.output/'summary.json')
    np.savez_compressed(args.output/'fields.npz',**fields)
    print('AGGREGATES',json.dumps(aggregates),flush=True)


if __name__=='__main__':main()
