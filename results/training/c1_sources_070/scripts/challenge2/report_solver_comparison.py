#!/usr/bin/env python3
"""Summarize completed MAST solver checks without making accuracy claims."""
from pathlib import Path
import argparse
import json
import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    args=parser.parse_args();out=args.directory
    summary=json.loads((out/'summary.json').read_text())
    data=np.load(out/'fields.npz')
    records=summary['records']
    comparisons=[]
    for method in ('anderson','krylov'):
        cold={r['frame_index']:r for r in records if r['group']=='segment_cold' and r['method']==method}
        warm={r['frame_index']:r for r in records if r['group']=='segment_continued' and r['method']==method}
        for frame in sorted(set(cold)&set(warm)):
            a,b=cold[frame],warm[frame]
            prefix=f'{method}_shot_0000_frame_{frame:03d}'
            pa=data['segment_cold_'+prefix+'_psi'].astype(float)
            pb=data['segment_continued_'+prefix+'_psi'].astype(float)
            record={'method':method,'frame':frame,'time_ms':a.get('time_ms'),
                'both_converged':bool(a['converged'] and b['converged']),
                'both_structurally_admissible':bool(a['structurally_admissible_equilibrium'] and b['structurally_admissible_equilibrium']),
                'flux_RMSE_Wb_per_rad':float(np.sqrt(np.mean((pa-pb)**2))),
                'q95_cold':a.get('q95'),'q95_continued':b.get('q95'),
                'cold_evaluations':a.get('function_evaluations'),'continued_evaluations':b.get('function_evaluations')}
            if a.get('derived') and b.get('derived'):
                record['axis_R_difference_m']=b['derived']['R_axis']-a['derived']['R_axis']
                record['axis_Z_difference_m']=b['derived']['Z_axis']-a['derived']['Z_axis']
            comparisons.append(record)
    checks={'max_Ip_relative_error':max(r.get('Ip_relative_error',0) for r in records),
        'max_linear_GS_relative_residual':max(r.get('linear_GS_relative_residual',0) for r in records),
        'native_solution_comparisons':comparisons,
        'qualification':'Both small residual and independent structural checks are necessary; none establishes hidden MAST accuracy.'}
    (out/'solution_agreement.json').write_text(json.dumps(checks,indent=2,allow_nan=False)+'\n')
    lines=['# Fixed-coil MAST nonlinear solver comparison','',
        'Public-input physics diagnostics only: no MAST, demo, hidden, or audit labels; no official score or upload.','',
        'All methods use the same actual coil geometry, measured currents, TS pressure closure, and free-space gauge. '
        'Convergence requires the recomputed relative current-profile residual below the recorded tolerance. '
        'Structural admissibility additionally requires an independent LCFS-adjacent saddle, no envelope contact, '
        'negligible negative current, and valid q95/betaN/F².','',
        '| Group | Method | Converged | Structurally admissible | Scalar valid | Evaluations | Solver seconds |',
        '|---|---|---:|---:|---:|---:|---:|']
    for a in summary['aggregates']:
        lines.append(f"| {a['group']} | {a['method']} | {a['converged']}/{a['frames']} | "
                     f"{a['structurally_admissible']}/{a['frames']} | {a['scalar_valid']}/{a['frames']} | "
                     f"{a['total_function_evaluations']} | {a['total_solver_seconds']:.2f} |")
    lines+=['',f"Maximum current-integral relative error: {checks['max_Ip_relative_error']:.3g}; "
             f"maximum linear GS relative residual: {checks['max_linear_GS_relative_residual']:.3g}.",'',
             'Cold and continued solutions are compared in `solution_agreement.json`. Agreement at the native grid '
             'and at a stricter residual tolerance should be checked before interpreting differences as distinct '
             'equilibrium branches. TS-pressure mismatch remains a closure-model diagnostic.','',
             'The closure and the absolute EFIT gauge remain unvalidated against MAST labels. '
             'This bounded run does not demonstrate full-shot coverage or submission readiness.']
    (out/'README.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(14,4),constrained_layout=True)
    main=[a for a in summary['aggregates'] if a['group']=='same_frames']
    x=np.arange(len(main))
    axes[0].bar(x-.17,[a['converged'] for a in main],.34,label='Closure converged')
    axes[0].bar(x+.17,[a['structurally_admissible'] for a in main],.34,label='Also passes structure')
    axes[0].set(xticks=x,xticklabels=[a['method'] for a in main],ylabel='Frames',title='Same 12 native frames',ylim=(0,12.7))
    axes[0].legend(fontsize=8)
    for method,color in [('anderson','C1'),('krylov','C2')]:
        for group,style in [('segment_cold','--'),('segment_continued','-')]:
            part=[r for r in records if r['method']==method and r['group']==group]
            if not part:continue
            times=[r.get('time_ms') for r in part]
            axes[1].plot(times,[r.get('q95') for r in part],style,color=color,marker='o',ms=3,label=method+' '+group.removeprefix('segment_'))
            axes[2].plot(times,[r.get('function_evaluations') for r in part],style,color=color,marker='o',ms=3)
    axes[1].set(xlabel='Native time [ms]',ylabel='Closure q95',title='Cold versus temporal continuation')
    axes[1].legend(fontsize=7)
    axes[2].set(xlabel='Native time [ms]',ylabel='Function evaluations',title='Continuation cost')
    fig.suptitle('MAST fixed-coil solver checks — reconstruction accuracy unknown')
    fig.savefig(out/'solver_comparison.png',dpi=170)
    plt.close(fig)
    print(json.dumps({k:v for k,v in checks.items() if k!='native_solution_comparisons'}))


if __name__=='__main__':main()
