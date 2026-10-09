"""Dimensionless GS profile targets and geometry-aware, input-only features.

Target decomposition is explicitly separate from deployable feature extraction.
The positive variables are p' and C=FF'/mu0+R_inner²*p', not FF' itself.
R_inner is the current plasma contour's inner radius, never the grid radius.
"""
from __future__ import annotations
from pathlib import Path
import hashlib
import re
import math
import numpy as np
from scipy.optimize import nnls
from scipy.ndimage import median_filter
from fusion.data import arrow_to_numpy,interpolate_signal
from fusion.mast_physics import (MU0,E_CHARGE,FreeBoundaryGS,ring_green,coil_response,_surface)
from starter.data_fixes import fix_d3d_ip_times,align_mast_thomson_core

ROOT=Path(__file__).resolve().parents[1]
DEGREE=3
N_BASIS=DEGREE+1
_GEOMETRY_CACHE={}


def read_row(path,labels=False):
    import pyarrow.parquet as pq
    names=pq.ParquetFile(path).schema_arrow.names
    exact={'source','efit_times','efit_grid_R','efit_grid_Z','magnetics_time',
           'magnetics_plasma_current_times','thomson_core_R','thomson_edge_spatial'}
    currents={'magnetics_ECOILA','magnetics_bcoil','magnetics_plasma_current','magnetics_sol_current',
              'magnetics_tf_current','magnetics_efps_current'}
    currents|={f'magnetics_F{n}{s}' for n in range(1,10) for s in 'AB'}
    currents|={f'magnetics_p{n}{s}_current' for n in range(2,7) for s in 'lu'}
    exact|=currents|{f'thomson_{system}_{field}' for system in ['core','edge'] for field in ['times','Te','ne']}
    columns=[k for k in names if k in exact or k.startswith(('coil_','thomson_chord_'))]
    if labels:
        if 'diii_d_train' not in str(path):raise ValueError('Labels may only be read from released DIII-D train paths')
        columns+=['efit_psirz','efit_q95','efit_beta_n']
    table=pq.read_table(path,columns=columns)
    row={k:arrow_to_numpy(table[k][0]) for k in columns}
    if labels and row['source']!='DIII-D':raise ValueError('MAST labels are forbidden')
    return row


def bernstein(s):
    s=np.clip(np.asarray(s,float),0,1)
    return np.stack([math.comb(DEGREE,k)*s**k*(1-s)**(DEGREE-k) for k in range(N_BASIS)],axis=-1)


def bernstein_integral(s):
    s=np.clip(np.asarray(s,float),0,1)
    values=[]
    for k in range(N_BASIS):
        total=np.zeros_like(s)
        for j in range(DEGREE-k+1):
            total+=math.comb(DEGREE,k)*math.comb(DEGREE-k,j)*(-1)**j*s**(k+j+1)/(k+j+1)
        values.append(total)
    return np.stack(values,axis=-1)


def length_scale(row):
    return float(row['efit_grid_R'][-1]-row['efit_grid_R'][0])


def current_at(row,times):
    timebase=fix_d3d_ip_times(row) if row['source']=='DIII-D' else row['magnetics_time']
    return interpolate_signal(timebase,row['magnetics_plasma_current'],np.asarray(times,float))


def machine_fields(row):
    geometry=np.stack([row[k] for k in ['coil_R','coil_Z','coil_width','coil_height']]).astype(float)
    fingerprint=hashlib.sha256(geometry.tobytes()+np.r_[row['efit_grid_R'],row['efit_grid_Z']].tobytes()+row['source'].encode()).hexdigest()
    if fingerprint in _GEOMETRY_CACHE:return _GEOMETRY_CACHE[fingerprint]
    R,Z=np.asarray(row['efit_grid_R'],float),np.asarray(row['efit_grid_Z'],float)
    RR,ZZ=np.meshgrid(R,Z)
    if row['source']=='MAST':
        response=coil_response(row)
        value={'columns':response.columns,'responses':response.psi_per_kilo_unit,
               'cs_key':'magnetics_sol_current','cs_unit':None,'geometry_sha256':fingerprint}
    else:
        columns=list(row['coil_input_column']);responses=[];cs_unit=None
        for i,(key,a,b) in enumerate(zip(columns,row['coil_R'],row['coil_Z'])):
            if key=='magnetics_ECOILA':
                width,height=geometry[2,i],geometry[3,i]
                xr,wr=np.polynomial.legendre.leggauss(8);xz,wz=np.polynomial.legendre.leggauss(32)
                field=np.zeros_like(RR)
                for u,w in zip(xr,wr):
                    G=ring_green(RR[None],ZZ[None],a+u*width/2,(b+xz*height/2)[:,None,None])
                    field+=w/2*np.einsum('n,nzr->zr',wz/2,G)
                cs_unit=field
            else:
                field=ring_green(RR,ZZ,float(a),float(b))
            responses.append(1000*field)
        value={'columns':columns,'responses':np.stack(responses),'cs_key':'magnetics_ECOILA',
               'cs_unit':cs_unit,'geometry_sha256':fingerprint}
    _GEOMETRY_CACHE[fingerprint]=value
    return value


def input_features(row,frame):
    """Never reads any target, inferred axis, boundary, or derived target quantity."""
    time=float(row['efit_times'][frame]);L=length_scale(row)
    signed_ip,ipmask=current_at(row,[time]);signed_ip=float(signed_ip[0])*1000
    ip=abs(signed_ip);safe_ip=max(ip,1e4);orientation=1. if signed_ip>=0 else -1.
    pressure_scale=MU0*safe_ip**2/L**2
    names=[];features=[]
    def add(key,value):names.append(key);features.append(float(value))
    add('log_Ip_MA',np.log1p(safe_ip/1e6));add('Ip_observed',ipmask[0])
    add('grid_Rmin_over_L',row['efit_grid_R'][0]/L)
    add('grid_Zspan_over_L',(row['efit_grid_Z'][-1]-row['efit_grid_Z'][0])/L)
    pressure_samples=[]
    for system in ['core','edge']:
        Te=np.asarray(row[f'thomson_{system}_Te'],float);ne=np.asarray(row[f'thomson_{system}_ne'],float)
        ts=np.asarray(row[f'thomson_{system}_times'],float)
        if Te.shape[0]!=len(ts):Te=Te.T;ne=ne.T
        if row['source']=='MAST' and system=='core':_,Te,ne=align_mast_thomson_core(row)
        channels=Te.shape[1]
        coords=np.zeros((channels,2));coordvalid=np.zeros(channels,bool)
        group='core' if system=='core' else 'tangential'
        for name,r,z in zip(row['thomson_chord_name'],row['thomson_chord_R'],row['thomson_chord_Z']):
            match=re.fullmatch(rf'TS_{group}_(?:.*_)?(\d+)',str(name))
            if match and int(match.group(1))<channels:
                index=int(match.group(1));coords[index]=(r,z);coordvalid[index]=True
        values=[];observed=[]
        for source in [Te,ne]:
            data=[];mask=[]
            for j in range(channels):
                v,m=interpolate_signal(ts,np.where(source[:,j]>=0,source[:,j],np.nan),np.array([time]))
                data.append(float(v[0]));mask.append(bool(m[0]))
            values.append(np.asarray(data));observed.append(np.asarray(mask))
        pressure=E_CHARGE*np.maximum(values[0],0)*np.maximum(values[1],0)
        pressure=median_filter(pressure,size=3,mode='nearest')
        pressure_samples.extend(pressure[pressure>0].tolist())
        for quantity,data,scale in [('Te',values[0],1000.),('ne',values[1],1e19),('pe',pressure,pressure_scale)]:
            positive=data[data>0]
            stats=np.quantile(positive,[.25,.5,.9,1]) if len(positive) else np.zeros(4)
            for q,value in zip(['p25','p50','p90','max'],stats):add(f'{system}_{quantity}_{q}',np.log1p(value/scale))
            add(f'{system}_{quantity}_positive_fraction',len(positive)/channels)
        add(f'{system}_observed_fraction',np.mean(observed[0]&observed[1]))
        weights=np.where(coordvalid,pressure,0)
        if weights.sum()>0:
            weights/=weights.sum();center=weights@coords/L
            centered=coords/L-center
            covariance=(centered*weights[:,None]).T@centered
            geom=[*center,covariance[0,0],covariance[1,1],covariance[0,1]]
        else:geom=[0.]*5
        for key,v in zip(['p_R','p_Z','p_RR','p_ZZ','p_RZ'],geom):add(f'{system}_{key}',v)
    pe_ref=float(np.quantile(pressure_samples,.9)) if pressure_samples else 0.
    coil=machine_fields(row)
    currents=[]
    for key in coil['columns']:
        current,_=interpolate_signal(row['magnetics_time'],row[key],np.array([time]))
        currents.append(float(current[0])*orientation)
    pieces=np.asarray(currents)[:,None,None]*coil['responses']
    cs_index=coil['columns'].index(coil['cs_key']);cs=pieces[cs_index];pf=pieces.sum(0)-cs
    for key,field in [('PF',pf),('CS_nominal',cs)]:
        sample=field[np.ix_([8,20,32,44,56],[8,20,32,44,56])]/(MU0*safe_ip*L)
        for j,value in enumerate(sample.ravel()):add(f'{key}_psi_star_{j:02d}',value)
    for offset in [-50,-10,10,50]:
        value,mask=current_at(row,[time+offset])
        add(f'Ip_relative_change_{offset:+d}ms',(float(value[0])*1000*orientation-ip)/safe_ip)
        add(f'Ip_observed_{offset:+d}ms',mask[0])
    result=np.asarray(features,float)
    if not np.isfinite(result).all():raise ValueError('Nonfinite input-only features')
    return result,{'feature_names':names,'time_ms':time,'L':L,'Ip_A':ip,'Ip_signed_A':signed_ip,
        'Ip_observed':bool(ipmask[0]),'pe_ref_Pa':pe_ref,'pressure_scale':pressure_scale,
        'pf_flux':pf,'cs_nominal_flux':cs,'cs_unit':coil['cs_unit'],'geometry_sha256':coil['geometry_sha256']}


def fit_profile_target(phi,solver,envelope,ip,L,regularization=1e-3):
    """Oracle target decomposition. Never call this to construct predictor inputs."""
    surface=_surface(phi,solver,envelope)
    if surface is None:raise ValueError('Target has no closed surface')
    C,axis,edge,delta,inside=surface
    if ip<1e4:raise ValueError('Observed Ip below10kA cannot normalize this diverted target reliably')
    current=-(solver.matrix@phi.ravel()).reshape(solver.shape)/(MU0*solver.RR)
    current[~inside]=0;current.ravel()[solver.boundary]=0
    s=np.clip((phi-edge)/delta,0,1)
    r=solver.RR/L;floor=float(C[:,0].min())/L
    B=bernstein(s)
    design=np.concatenate([(r-floor**2/r)[...,None]*B,B/r[...,None]],axis=-1)
    selected=inside&(s>.025)&(s<.975)
    if selected.sum()<32:raise ValueError('Too few resolved plasma points')
    shell=np.minimum((s[selected]*12).astype(int),11)
    weights=1/np.sqrt(np.bincount(shell,minlength=12)[shell])
    X=design[selected];y=current[selected]*L**2/ip
    Xw=X*weights[:,None];yw=y*weights
    scale=np.sqrt(np.sum(Xw**2,axis=0));scale=np.maximum(scale,1e-8)
    XX=Xw/scale
    penalty=np.sqrt(regularization)*np.eye(2*N_BASIS)
    augmented=np.vstack([XX,penalty]);rhs=np.r_[yw,np.zeros(2*N_BASIS)]
    transformed,nnls_residual=nnls(augmented,rhs,maxiter=1000)
    coefficients=transformed/scale
    unconstrained=np.linalg.lstsq(augmented,rhs,rcond=None)[0]/scale
    grid_floor=solver.R[0]/L
    grid_design=np.concatenate([(r-grid_floor**2/r)[...,None]*B,B/r[...,None]],axis=-1)
    grid_Xw=grid_design[selected]*weights[:,None]
    grid_scale=np.maximum(np.linalg.norm(grid_Xw,axis=0),1e-8)
    grid_coef=nnls(np.vstack([grid_Xw/grid_scale,penalty]),rhs,maxiter=1000)[0]/grid_scale
    grid_relative_error=np.linalg.norm(grid_design[selected]@grid_coef-y)/max(np.linalg.norm(y),1e-30)
    pred=np.einsum('zrk,k->zr',design,coefficients)*ip/L**2
    pred[~inside]=0
    source_error=float(np.linalg.norm((pred-current)[selected])/max(np.linalg.norm(current[selected]),1e-30))
    a,c=coefficients[:N_BASIS],coefficients[N_BASIS:]
    b=c-floor**2*a
    paxis_star=delta/(MU0*ip*L)*a.sum()/N_BASIS
    halfwidth=.5*np.ptp(C[:,0]);halfheight=.5*np.ptp(C[:,1])
    # Split poloidal angles, retaining all flux shells, to expose separation instability.
    angular=solver.ZZ[selected]>=axis[1]
    refits=[]
    for subset in [angular,~angular]:
        Ahalf=XX[subset];bhalf=yw[subset]
        if len(bhalf)<2*N_BASIS:continue
        sol,_=nnls(np.vstack([Ahalf,penalty]),np.r_[bhalf,np.zeros(2*N_BASIS)],maxiter=1000)
        refits.append(sol/scale)
    angular_instability=float(np.linalg.norm(refits[0]-refits[1])/max(np.linalg.norm(coefficients),1e-10)) if len(refits)==2 else None
    return {'coefficients':coefficients,'unconstrained_coefficients':unconstrained,
        'A_shape':a/max(a.sum(),1e-12),'C_shape':c/max(c.sum(),1e-12),
        'paxis_star':paxis_star,'geometry':np.array([axis[0]/L,axis[1]/L,halfwidth/L,halfheight/max(halfwidth,1e-8)]),
        'oracle_current':current,'fitted_current':pred,'inside':inside,'s':s,
        'condition_number':float(np.linalg.cond(XX)),'angular_split_relative_instability':angular_instability,
        'R_inner_over_L':floor,'grid_floor_over_L':grid_floor,
        'grid_floor_source_relative_RMSE':float(grid_relative_error),
        'grid_floor_pressure_axis_Pa':float(delta*ip/L**3*grid_coef[:N_BASIS].sum()/N_BASIS),
        'unconstrained_pressure_axis_Pa':float(delta*ip/L**3*unconstrained[:N_BASIS].sum()/N_BASIS),
        'positive_bound_active_fraction':float(np.mean(coefficients<=1e-10)),
        'grid_floor_bound_active_fraction':float(np.mean(grid_coef<=1e-10)),
        'unconstrained_A_min':float(unconstrained[:N_BASIS].min()),
        'unconstrained_current_min':float((design@unconstrained)[inside].min()*ip/L**2),
        'source_relative_RMSE':source_error,'unconstrained_source_relative_RMSE':float(np.linalg.norm(X@unconstrained-y)/max(np.linalg.norm(y),1e-30)),
        'Ip_from_target_A':float(current.sum()*solver.dR*solver.dZ),'input_Ip_ratio':float(current.sum()*solver.dR*solver.dZ/ip),
        'A_min':float(a.min()),'B_min':float(b.min()),'C_min':float(c.min()),
        'fitted_current_min':float(pred[inside].min()),'pressure_axis_Pa':paxis_star*MU0*ip**2/L**2,
        'edge_flux':edge,'delta_flux':delta}


def learned_current(phi,solver,envelope,ip,paxis,A_shape,C_shape):
    surface=_surface(phi,solver,envelope)
    if surface is None:raise ValueError('no_closed_plasma_surface')
    contour,_,edge,delta,inside=surface
    basis=bernstein(np.clip((phi-edge)/delta,0,1))
    Ashape=np.asarray(A_shape,float);Cshape=np.asarray(C_shape,float)
    if np.min(Ashape)<0 or np.min(Cshape)<0:raise ValueError('Positive shape coefficients required')
    ashape=basis@Ashape/(max(Ashape.sum(),1e-12)/N_BASIS)
    cshape=basis@Cshape/max(Cshape.sum(),1e-12)
    pprime=paxis/delta*ashape
    inner_radius=float(contour[:,0].min())
    pressure_current=(solver.RR-inner_radius**2/solver.RR)*pprime*inside
    density_shape=cshape/solver.RR*inside
    amplitude=(ip-pressure_current.sum()*solver.dR*solver.dZ)/(density_shape.sum()*solver.dR*solver.dZ)
    if amplitude<0:raise ValueError('Predicted pressure requires a negative transformed-current amplitude')
    return pressure_current+amplitude*density_shape
