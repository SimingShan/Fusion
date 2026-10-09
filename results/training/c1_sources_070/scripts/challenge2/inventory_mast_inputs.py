#!/usr/bin/env python3
"""Inventory signed Ip and native-grid coverage using MAST public inputs only."""
from pathlib import Path
import hashlib
import json
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
import pyarrow.parquet as pq
from fusion.data import arrow_to_numpy,interpolate_signal


def main():
    out=ROOT/'results/challenge2/input_inventory_v1'
    out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((ROOT/'data/raw/download_manifest.json').read_text())
    files=sorted(x['path'] for x in manifest['files'] if x['path'].startswith('data/mast_public_test/')
                 and x['path'].endswith('.parquet'))
    columns=['source','efit_times','efit_grid_R','efit_grid_Z','magnetics_time','magnetics_plasma_current',
             'thomson_core_times','coil_input_column','coil_R','coil_Z','coil_width','coil_height']
    records=[];all_ip=[];native={}
    for i,file in enumerate(files):
        table=pq.read_table(ROOT/'data/raw'/file,columns=columns)
        row={k:arrow_to_numpy(table[k][0]) for k in columns}
        assert row['source']=='MAST'
        times=np.asarray(row['efit_times'],float)
        ip,observed=interpolate_signal(row['magnetics_time'],row['magnetics_plasma_current'],times)
        all_ip.append(ip)
        native[f'shot_{i:04d}_Ip_kA']=ip
        native[f'shot_{i:04d}_times_ms']=times
        geometry=np.stack([np.asarray(row[k],float) for k in ['coil_R','coil_Z','coil_width','coil_height']])
        ts=np.asarray(row['thomson_core_times'],float);ts=ts[np.isfinite(ts)]
        records.append({'canonical_shot_index':i,'source':file,'frames':len(times),
            'Ip_min_kA':float(ip.min()),'Ip_max_kA':float(ip.max()),'negative_Ip_frames':int((ip<0).sum()),
            'near_zero_Ip_frames_abs_below10kA':int((np.abs(ip)<10).sum()),'unobserved_Ip_frames':int((observed==0).sum()),
            'Ip_native_finite_fraction':float(np.isfinite(row['magnetics_plasma_current']).mean()),
            'frames_outside_core_TS_time_interval':int(((times<ts.min())|(times>ts.max())).sum()) if len(ts) else len(times),
            'grid_shape':[len(row['efit_grid_Z']),len(row['efit_grid_R'])],
            'grid_sha256':hashlib.sha256(np.r_[row['efit_grid_R'],row['efit_grid_Z']].tobytes()).hexdigest(),
            'geometry_sha256':hashlib.sha256(geometry.tobytes()+'|'.join(row['coil_input_column']).encode()).hexdigest()})
        if (i+1)%200==0:print('Inventoried',i+1,'/',len(files),flush=True)
    ip=np.concatenate(all_ip)
    output={'purpose':'Public-input signed-current and coverage inventory before solver deployment',
        'label_role':'No MAST/demo/hidden/audit labels','revision':manifest['revision'],'shots':len(records),
        'total_native_frames':len(ip),'negative_Ip_frames':int((ip<0).sum()),
        'near_zero_Ip_frames_abs_below10kA':int((np.abs(ip)<10).sum()),
        'unobserved_Ip_frames':sum(r['unobserved_Ip_frames'] for r in records),
        'Ip_quantiles_kA':dict(zip(['min','p01','p50','p99','max'],np.quantile(ip,[0,.01,.5,.99,1]).tolist())),
        'unique_grid_count':len(set(r['grid_sha256'] for r in records)),
        'unique_geometry_count':len(set(r['geometry_sha256'] for r in records)),
        'records':records}
    np.savez_compressed(out/'native_inputs.npz',**native)
    (out/'summary.json').write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in output.items() if k!='records'}),flush=True)


if __name__=='__main__':main()
