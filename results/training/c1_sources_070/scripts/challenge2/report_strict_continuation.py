#!/usr/bin/env python3
"""Create a compact report of an already completed strict native continuation."""
from pathlib import Path
import argparse,json
import numpy as np

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path)
    args=p.parse_args();out=args.directory
    report=json.loads((out/'summary.json').read_text());rows=report['records']
    fields=np.load(out/'fields.npz');old=np.load(Path(report['source'])/'fields.npz')
    summary={'frames':len(rows),'converged':sum(r['converged'] for r in rows),
        'structurally_admissible':sum(r['structurally_admissible_equilibrium'] for r in rows),
        'scalar_valid':sum(r['scalar_valid'] for r in rows),
        'max_profile_relative_residual':max(r['nonlinear_profile_relative_residual'] for r in rows),
        'max_Ip_relative_error':max(r['Ip_relative_error'] for r in rows),
        'max_linear_GS_relative_residual':max(r['linear_GS_relative_residual'] for r in rows),
        'function_evaluations':sum(r['function_evaluations'] for r in rows),
        'solver_seconds':sum(r['solver_elapsed_seconds'] for r in rows),
        'q95_range':[min(r['q95'] for r in rows),max(r['q95'] for r in rows)],
        'betaN_range':[min(r['betaN'] for r in rows),max(r['betaN'] for r in rows)],
        'TS_pressure_relative_RMSE_range':[min(r['TS_pressure_relative_RMSE'] for r in rows),max(r['TS_pressure_relative_RMSE'] for r in rows)],
        'time_range_ms':[min(r['time_ms'] for r in rows),max(r['time_ms'] for r in rows)],
        'accuracy_claim':'None: these are public-input numerical and structural checks, without MAST labels',
        'initialization':'First frame seeded by the documented loose numerical solution; every subsequent frame continues from the preceding strictly converged source'}
    (out/'checks.json').write_text(json.dumps(summary,indent=2)+'\n')
    lines=['# Strict fixed-coil native continuation','',
        f"**{summary['converged']}/{summary['frames']} frames converge and {summary['structurally_admissible']}/{summary['frames']} pass independent structural checks** at profile tolerance 1e−4.",'',
        'The segment is MAST public-input shot0, native frames18–25 (190–225ms). '
        'No MAST/demo/hidden/audit labels were used. No coil, gauge or pressure parameter was fitted in this comparison. '
        'The first frame is seeded from its previous loose solve; subsequent frames use the preceding strict source, rescaled to measured Ip.','',
        f"Maximum profile residual **{summary['max_profile_relative_residual']:.3g}**; current integral error "
        f"**{summary['max_Ip_relative_error']:.3g}**; linear GS residual **{summary['max_linear_GS_relative_residual']:.3g}**. "
        f"The eight solves use {summary['function_evaluations']} evaluations and {summary['solver_seconds']:.2f} CPU seconds.",'',
        'Independent bicubic-gradient roots confirm LCFS-adjacent saddle points, '
        'the extracted surface does not contact the published envelope cap, current remains nonnegative, '
        'and q95/betaN/F² are finite and valid under the assumed closure.','',
        'A separate tolerance experiment showed that some apparently converged 0.002-residual states '
        'were insufficiently accurate: at frame22 a cold Newton solution moved from q95=3.738 to 3.0228 '
        'at 1e−4, agreeing with a continued solution at 3.0243. Therefore 0.002 is unsuitable as a final acceptance threshold.','',
        'This establishes a working numerical solver on one short segment. It does not establish MAST reconstruction accuracy, '
        'unique equilibria for all inputs, full-shot coverage, or a correct EFIT gauge. '
        'Electron-pressure mismatch and the assumed ion/FF′ profiles still motivate a training-only DIII-D closure prior.']
    (out/'README.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(13,4),constrained_layout=True)
    row=rows[len(rows)//2];key=row['initialization_source']
    R,Z=old[key+'_R'],old[key+'_Z'];psi=fields[key+'_psi']
    image=axes[0].contourf(R,Z,psi,levels=30,cmap='RdBu_r')
    axes[0].contour(R,Z,psi,levels=30,colors='k',linewidths=.4)
    saddles=[s for s in row['structure']['saddles'] if s['supports_LCFS_separatrix']]
    for s in saddles:axes[0].plot(s['R'],s['Z'],'x',color='lime',ms=8,mew=2)
    axes[0].set(xlabel='R [m]',ylabel='Z [m]',title=f"Fixed-coil flux, {row['time_ms']:.0f} ms",xlim=(.06,1.8),ylim=(-1.7,1.7))
    axes[0].set_aspect('equal');fig.colorbar(image,ax=axes[0],label='ψ [Wb/rad]')
    t=[r['time_ms'] for r in rows]
    axes[1].semilogy(t,[r['nonlinear_profile_relative_residual'] for r in rows],'o-')
    axes[1].axhline(1e-4,color='k',ls='--',label='Strict tolerance')
    axes[1].set(xlabel='Native time [ms]',ylabel='Relative profile residual',title='8/8 strict native solves')
    axes[1].legend(fontsize=8)
    axes[2].plot(t,[r['q95'] for r in rows],'o-',label='q95')
    axes[2].plot(t,[r['betaN'] for r in rows],'s-',label='betaN')
    axes[2].set(xlabel='Native time [ms]',ylabel='Scalar under assumed closure',title='Finite physics-derived outputs')
    axes[2].legend(fontsize=8)
    fig.suptitle('MAST numerical feasibility — experimental accuracy remains unmeasured')
    fig.savefig(out/'strict_continuation.png',dpi=180);plt.close(fig)
    print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
