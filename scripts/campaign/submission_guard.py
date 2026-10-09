"""Enforce the user's complete-batch/top-three submission policy."""
import json
import hashlib
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def is_terminal_outcome(row):
    if row.get('status') == 'complete':
        return True
    if row.get('status') not in {'failed', 'infeasible', 'budget_exhausted', 'timed_out', 'cancelled'}:
        return False
    review = row.get('failure_review', {})
    return (review.get('terminal_method_outcome') is True and bool(review.get('reason'))
            and isinstance(review.get('cost_seconds'), (int, float)) and review['cost_seconds'] >= 0)


def require_batch_selection(root, run, archive_sha256, phase):
    policy_path = root / 'results/analysis/daily_campaign_policy.json'
    if phase != 29580 or not policy_path.exists():
        return
    policy = read(policy_path)
    if not policy.get('active'):
        return
    if (root / 'scripts/campaign/PAUSE').exists():
        raise RuntimeError('The daily campaign is paused; no new submission is allowed.')
    state = read(root / 'results/analysis/daily_campaign_state.json')
    batch = root / 'results/analysis' / ('daily_campaign_' + state['current_batch'])
    results_path, selection_path = batch / 'results.json', batch / 'selection.json'
    if not results_path.exists() or not selection_path.exists():
        raise RuntimeError('Daily policy requires all20 terminal outcomes and a saved top-three selection before submission.')
    results, selection = read(results_path), read(selection_path)
    expected = {str(x['experiment']) for x in read(batch / 'plan.json')['experiments']}
    outcomes = results.get('experiments', {})
    if (len(expected) != 20 or set(outcomes) != expected or not results.get('selection_allowed')
            or not all(is_terminal_outcome(row) for row in outcomes.values())):
        raise RuntimeError('The daily20-method batch has unresolved outcomes.')
    for name in ('results', 'plan', 'baseline'):
        if selection.get(name + '_sha256') != sha(batch / (name + '.json')):
            raise RuntimeError('The shortlist is not bound to the current batch evidence.')
    candidates = selection.get('candidates', [])
    if (selection.get('status') != 'final' or selection.get('batch') != state['current_batch']
            or not 1 <= len(candidates) <= min(3, policy['max_submissions_per_batch'])
            or len({x['candidate'] for x in candidates}) != len(candidates)):
        raise RuntimeError('The daily shortlist is not a final unique top-three selection.')
    chosen = next((x for x in candidates if x['candidate'] == run.name), None)
    if chosen is None or chosen.get('archive_sha256') != archive_sha256:
        raise RuntimeError('This exact candidate archive is absent from the completed batch shortlist.')
    report_path = root / chosen['source_report_path']
    if sha(report_path) != chosen.get('source_report_sha256'):
        raise RuntimeError('The chosen development report changed after selection.')
    if chosen.get('source_experiment') == 150 and run.name == '20261006-084':
        if report_path.resolve() != (root / 'results/analysis/geometry_reconcile150/evaluation/report.json').resolve():
            raise RuntimeError('Unexpected carry-over evidence for084.')
    else:
        outcome = outcomes.get(str(chosen.get('source_experiment')), {})
        if outcome.get('status') != 'complete' or Path(outcome['report_path']).resolve() != report_path.resolve():
            raise RuntimeError('Selected candidate is not a complete endpoint in this batch.')
        if (outcome['comparisons']['native']['S'] < 0 or outcome['comparisons']['legacy']['S'] < -.0001
                or any(value for fold in outcome['prediction_failures'].values() for value in fold.values())):
            raise RuntimeError('Selected candidate does not meet the development score/prediction guard.')
