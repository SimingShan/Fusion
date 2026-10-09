"""Label-free structural checks on a reconstructed MAST equilibrium."""
from __future__ import annotations

import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.optimize import root


def check_separatrix(psi, R, Z, contour, edge, delta, row):
    """Find independent spline critical points, then test LCFS/saddle agreement.

    A closed contour alone does not demonstrate a diverted separatrix. A saddle
    must be outside active conductors, near the extracted boundary, and have
    nearly the same flux. Tolerances reflect the native grid and are reported.
    This check cannot establish accuracy against any unobserved equilibrium.
    """
    psi=np.asarray(psi,float);R=np.asarray(R,float);Z=np.asarray(Z,float)
    spline=RectBivariateSpline(Z,R,psi,kx=3,ky=3,s=0)
    dz,dr=np.gradient(psi,Z,R)
    def cell_zero_span(a):
        corners=np.stack([a[:-1,:-1],a[1:,:-1],a[:-1,1:],a[1:,1:]])
        return (corners.min(0)<=0)&(corners.max(0)>=0)
    possible=cell_zero_span(dr)&cell_zero_span(dz)
    candidates=np.argwhere(possible)
    critical=[];saddles=[]
    grid_diag=float(np.hypot(np.mean(np.diff(R)),np.mean(np.diff(Z))))
    gradient_scale=max(float(np.ptp(psi))/(R[-1]-R[0]),1e-8)
    coil_R=np.asarray(row['coil_R']);coil_Z=np.asarray(row['coil_Z'])
    coil_w=np.asarray(row['coil_width']);coil_h=np.asarray(row['coil_height'])
    def gradient(x):
        r,z=x
        return np.array([spline.ev(z,r,dx=0,dy=1),spline.ev(z,r,dx=1,dy=0)])
    def hessian(x):
        r,z=x
        cross=spline.ev(z,r,dx=1,dy=1)
        return np.array([[spline.ev(z,r,dx=0,dy=2),cross],
                         [cross,spline.ev(z,r,dx=2,dy=0)]])
    for iz,ir in candidates[:128]:
        guess=np.array([0.5*(R[ir]+R[ir+1]),0.5*(Z[iz]+Z[iz+1])])
        answer=root(gradient,guess,jac=hessian,method='hybr',options={'maxfev':40,'xtol':1e-8})
        r,z=answer.x
        if not R[0]<r<R[-1] or not Z[0]<z<Z[-1]:continue
        gnorm=float(np.linalg.norm(gradient(answer.x)))
        if gnorm/gradient_scale>1e-5:continue
        if any(np.linalg.norm(answer.x-np.array([x['R'],x['Z']]))<0.25*grid_diag for x in critical):continue
        H=hessian(answer.x);det=float(np.linalg.det(H))
        point={'R':float(r),'Z':float(z),'psi':float(spline.ev(z,r)),
               'gradient_relative_norm':gnorm/gradient_scale,'Hessian_determinant':det,
               'kind':'saddle' if det<0 else ('maximum' if np.trace(H)<0 else 'minimum')}
        critical.append(point)
        if det>=0:continue
        # Coil centers contain their own magnetic extrema/saddles. Reject a
        # margin of one local grid diagonal around any supplied rectangle.
        dr_coil=np.maximum(np.abs(coil_R-r)-coil_w/2,0)
        dz_coil=np.maximum(np.abs(coil_Z-z)-coil_h/2,0)
        coil_distance=float(np.sqrt(dr_coil**2+dz_coil**2).min())
        distance=float(np.linalg.norm(contour-answer.x,axis=1).min())
        relative_flux_gap=abs(point['psi']-edge)/max(delta,1e-12)
        point.update({'distance_to_LCFS_m':distance,'relative_flux_gap_to_LCFS':relative_flux_gap,
                      'distance_to_active_conductor_m':coil_distance,
                      'supports_LCFS_separatrix': bool(distance<2*grid_diag and relative_flux_gap<0.03
                                                     and coil_distance>grid_diag)})
        saddles.append(point)
    return {'critical_points':critical,'saddles':saddles,
            'separatrix_supported':any(x['supports_LCFS_separatrix'] for x in saddles),
            'candidate_seed_count':len(candidates),'candidate_seed_limit':128,
            'distance_tolerance_m':2*grid_diag,'relative_flux_tolerance':0.03,
            'method':'Independent bicubic-spline gradient roots and Hessian saddle classification; native-grid tolerances'}
