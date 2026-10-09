"""Variable-size authorized cohorts share one terminal barrier and top-three limit."""
import json
from pathlib import Path
from scripts.campaign.submission_guard import (
    read, sha, is_terminal_outcome, require_batch_selection as legacy_require_batch_selection,
)


def extension_cohorts(root, state):
    """State extensions are explicit, unique {batch,path} records for current_batch."""
    cohorts=[]
    for entry in state.get('extensions', []):
        if not isinstance(entry,dict) or not entry.get('batch') or not entry.get('path'):
            raise RuntimeError('Malformed registered extension cohort')
        if entry.get('parent_batch',state['current_batch'])!=state['current_batch']:
            raise RuntimeError('Extension is registered under a different daily batch')
        path=(root/entry['path']).resolve()
        expected=(root/'results/analysis'/('daily_campaign_'+entry['batch'])).resolve()
        if path!=expected or entry['batch']==state['current_batch']:
            raise RuntimeError('Unexpected extension cohort path')
        cohorts.append((entry['batch'],path))
    if len({name for name,_ in cohorts})!=len(cohorts):raise RuntimeError('Duplicate extension cohort')
    return cohorts


def joint_evidence(root, state):
    """Read saved results only; missing/unreviewed outcomes keep selection closed."""
    main=root/'results/analysis'/('daily_campaign_'+state['current_batch'])
    cohorts=[(state['current_batch'],main)]+extension_cohorts(root,state)
    base_digest=sha(main/'baseline.json');all_outcomes={};bindings={}
    policy=read(root/'results/analysis/daily_campaign_policy.json')
    declared_counts=policy.get('cohort_experiment_counts',{})
    for index,(name,path) in enumerate(cohorts):
        plan=read(path/'plan.json');results=read(path/'results.json')
        expected={str(row['experiment']) for row in plan['experiments']}
        count=declared_counts.get(name)
        if type(count) is not int or count<1:
            raise RuntimeError('Cohort lacks an explicit positive policy experiment count')
        if len(expected)!=count or len(plan['experiments'])!=count:
            raise RuntimeError('Frozen cohort differs from explicitly authorized experiment count')
        if plan.get('expected_experiments',count)!=count:
            raise RuntimeError('Plan and policy cohort counts differ')
        if index and (plan.get('parent_batch')!=state['current_batch'] or sha(path/'baseline.json')!=base_digest):
            raise RuntimeError('Extension parent or copied baseline differs')
        outcomes=results.get('experiments',{})
        if set(outcomes)!=expected or not results.get('selection_allowed') or not all(is_terminal_outcome(row) for row in outcomes.values()):
            raise RuntimeError('Main batch or extension has unresolved outcomes')
        if set(all_outcomes)&expected:raise RuntimeError('Experiment IDs overlap between cohorts')
        for key,row in outcomes.items():all_outcomes[key]={**row,'cohort':name}
        bindings[name]={key+'_sha256':sha(path/(key+'.json')) for key in ('results','plan','baseline')}
    return all_outcomes,bindings


def eligible(row):
    return (row.get('status')=='complete'
        and row['comparisons']['native']['S']>=0
        and row['comparisons']['legacy']['S']>=-.0001
        and not any(value for fold in row['prediction_failures'].values() for value in fold.values()))


def require_batch_selection(root, run, archive_sha256, phase):
    policy_path=root/'results/analysis/daily_campaign_policy.json'
    if phase!=29580 or not policy_path.exists():
        return legacy_require_batch_selection(root,run,archive_sha256,phase)
    policy=read(policy_path)
    if not policy.get('active'):return legacy_require_batch_selection(root,run,archive_sha256,phase)
    state=read(root/'results/analysis/daily_campaign_state.json')
    registered={name for name,_ in extension_cohorts(root,state)}
    authorized=set(policy.get('authorized_extensions',{}).get(state['current_batch'],[]))
    if registered!=authorized:
        raise RuntimeError('Authorized extension is absent from state, or an unapproved extension was registered')
    if not registered:
        return legacy_require_batch_selection(root,run,archive_sha256,phase)
    if (root/'scripts/campaign/PAUSE').exists():raise RuntimeError('Daily campaign is paused')
    outcomes,bindings=joint_evidence(root,state)
    main=root/'results/analysis'/('daily_campaign_'+state['current_batch'])
    selection=read(main/'selection.json')
    for key,value in bindings[state['current_batch']].items():
        if selection.get(key)!=value:raise RuntimeError('Selection main evidence changed')
    extension_bindings={name:value for name,value in bindings.items() if name!=state['current_batch']}
    if selection.get('extensions')!=extension_bindings:raise RuntimeError('Selection extension evidence changed or missing')
    candidates=selection.get('candidates',[])
    if (selection.get('status')!='final' or selection.get('batch')!=state['current_batch']
            or not 1<=len(candidates)<=min(3,policy['max_submissions_per_batch'])
            or len({row['candidate'] for row in candidates})!=len(candidates)):
        raise RuntimeError('Joint shortlist must contain at most three unique final candidates')
    for chosen in candidates:
        report_path=(root/chosen['source_report_path']).resolve()
        if sha(report_path)!=chosen.get('source_report_sha256'):raise RuntimeError('Selected report changed')
        identifier=chosen.get('source_experiment')
        if identifier==150 and chosen['candidate']=='20261006-084':
            if report_path!=(root/'results/analysis/geometry_reconcile150/evaluation/report.json').resolve():
                raise RuntimeError('Unexpected084 carry-over source')
        else:
            row=outcomes.get(str(identifier),{})
            if row.get('status')!='complete' or (root/row['report_path']).resolve()!=report_path or not eligible(row):
                raise RuntimeError('Selected candidate is not an eligible completed joint endpoint')
            report=read(report_path)
            if report.get('status')!='complete' or report.get('experiment')!=identifier:
                raise RuntimeError('Selected report identity differs')
            if row.get('report_sha256') and sha(report_path)!=row['report_sha256']:
                raise RuntimeError('Collated selected report changed')
    chosen=next((row for row in candidates if row['candidate']==run.name),None)
    if chosen is None or chosen.get('archive_sha256')!=archive_sha256:
        raise RuntimeError('Exact archive is absent from the joint shortlist')
