#!/usr/bin/env python3
"""Native DIII-D source-profile targets; labels are isolated from input features."""
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
from fusion.mast_profiles import read_row,input_features,fit_profile_target,FreeBoundaryGS,MU0


def clean(value):
    if isinstance(value,dict):return {k:clean(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [clean(v) for v in value]
    if isinstance(value,np.ndarray):return clean(value.tolist())
    if isinstance(value,np.generic):return clean(value.item())
    if isinstance(value,float) and not np.isfinite(value):return None
    return value


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train-shots',type=int,default=128)
    p.add_argument('--fresh-shots',type=int,default=16)
    p.add_argument('--legacy-shots',type=int,default=8)
    p.add_argument('--train-frames',type=int,default=8)
    p.add_argument('--val-frames',type=int,default=4)
    p.add_argument('--output',type=Path,default=ROOT/'results/challenge2/profile_prior_v1')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    if (a.output/'targets.npz').exists():raise ValueError('Use a fresh output directory')
    manifest_path=ROOT/'data/advanced_v1/manifest.json'
    manifest=json.loads(manifest_path.read_text())
    with np.load(ROOT/'starter/fusion_scoring/masks/d3d_envelope.npz') as d:envelope=d['mask_coarse'].astype(bool)
    data={k:[] for k in ['X','phi','oracle_plasma','pf_flux','inside','A_shape','C_shape','paxis_star',
                         'geometry','Ip_A','L','pe_ref_Pa','fit_valid','shot_index','frame_index','fold']}
    records=[];started=time.monotonic();solver=None;feature_names=None;poison_check=None
    for fold,count,nframes in [('train',a.train_shots,a.train_frames),('val',a.fresh_shots,a.val_frames),
                                ('legacy_val',a.legacy_shots,a.val_frames)]:
        pool=manifest[fold]
        selected=[pool[i] for i in np.unique(np.linspace(0,len(pool)-1,count).astype(int))] if count else []
        for ordinal,entry in enumerate(selected):
            path=ROOT/'data/raw'/entry['shard'];row=read_row(path,labels=True)
            if solver is None:
                solver=FreeBoundaryGS(row['efit_grid_R'],row['efit_grid_Z'])
            if not (np.array_equal(solver.R,row['efit_grid_R']) and np.array_equal(solver.Z,row['efit_grid_Z'])):
                raise ValueError('This bounded experiment requires the common DIII-D grid')
            for frame in np.unique(np.linspace(0,len(row['efit_times'])-1,nframes).astype(int)):
                # This call has no target-derived arguments. Poison parity proves that
                # even present target fields do not influence extracted features.
                X,context=input_features(row,int(frame))
                if poison_check is None:
                    poisoned=dict(row);poisoned['efit_psirz']=None;poisoned['efit_q95']='forbidden';poisoned['efit_beta_n']=object()
                    X_poison,_=input_features(poisoned,int(frame))
                    poison_check=bool(np.array_equal(X,X_poison))
                    if not poison_check:raise AssertionError('Targets leaked into input features')
                if feature_names is None:feature_names=context['feature_names'];cs_unit=context['cs_unit']
                if feature_names!=context['feature_names']:raise ValueError('Feature schema changed')
                phi=-np.asarray(row['efit_psirz'][frame],float)
                record={'fold':fold,'shot_index':entry['shot_index'],'frame_index':int(frame),
                        'source':str(path.relative_to(ROOT)),'time_ms':context['time_ms'],
                        'label_role':'supervised training target' if fold=='train' else 'development diagnostic target, never predictor input'}
                try:
                    target=fit_profile_target(phi,solver,envelope,context['Ip_A'],context['L'])
                    plasma=solver.solve(target['oracle_current']);valid=True
                    record.update({k:v for k,v in target.items() if not isinstance(v,np.ndarray)})
                except (ValueError,RuntimeError,np.linalg.LinAlgError) as e:
                    target={'inside':np.zeros(solver.shape,bool),'A_shape':np.zeros(4),'C_shape':np.zeros(4),
                            'paxis_star':0.,'geometry':np.zeros(4)}
                    plasma=np.zeros(solver.shape);valid=False;record['failure']=str(e)
                record['fit_valid']=valid;records.append(record)
                values={'X':X,'phi':phi,'oracle_plasma':plasma,'pf_flux':context['pf_flux'],
                        'Ip_A':context['Ip_A'],'L':context['L'],'pe_ref_Pa':context['pe_ref_Pa'],
                        'fit_valid':valid,'shot_index':entry['shot_index'],'frame_index':frame,'fold':fold}
                values.update({k:target[k] for k in ['inside','A_shape','C_shape','paxis_star','geometry']})
                for k,v in values.items():data[k].append(v)
            print(json.dumps({'fold':fold,'shot_ordinal':ordinal,'shot_index':entry['shot_index'],
                              'frames_done':len(records),'elapsed_seconds':round(time.monotonic()-started,1)}),flush=True)
    arrays={k:np.asarray(v) for k,v in data.items()}
    for k in ['X','phi','oracle_plasma','pf_flux','A_shape','C_shape','paxis_star','geometry','Ip_A','L','pe_ref_Pa']:
        arrays[k]=arrays[k].astype(np.float32)
    arrays.update(R=solver.R,Z=solver.Z,envelope=envelope,cs_unit=cs_unit,feature_names=np.asarray(feature_names))
    np.savez_compressed(a.output/'targets.partial.npz',**arrays)
    os.replace(a.output/'targets.partial.npz',a.output/'targets.npz')
    summary={'purpose':'Train-only closure supervision and separately labeled development diagnostics',
        'label_role':'No audit, MAST, demonstration, hidden or participant targets; held-out labels only evaluate predictions',
        'selection':'Uniform deterministic shot positions per manifest partition; uniform native frame indices',
        'target_normalization':'L=grid radial span, I=abs observed plasma current, phi=-DIII-D stored psi',
        'constraint':'pprime>=0 and C=FFprime/mu0+R_inner^2*pprime>=0 within current plasma LCFS; FFprime may be negative',
        'input_target_poison_parity':poison_check,'feature_names':feature_names,'records':records,
        'manifest_sha256':hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in [Path(__file__),ROOT/'fusion/mast_profiles.py']},
        'elapsed_seconds':time.monotonic()-started,'settings':vars(a)|{'output':str(a.output)}}
    (a.output/'preparation.json').write_text(json.dumps(clean(summary),indent=2,allow_nan=False)+'\n')
    print(json.dumps({'done':True,'frames':len(records),'valid':int(arrays['fit_valid'].sum()),
                      'elapsed_seconds':time.monotonic()-started}),flush=True)


if __name__=='__main__':main()
