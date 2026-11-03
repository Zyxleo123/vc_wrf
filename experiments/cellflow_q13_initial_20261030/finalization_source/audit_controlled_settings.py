"""Separate fixed-protocol parity from numerical trajectory reproducibility.

This audit does not modify or retrain a model. The initial strict loss-trajectory
check remains in the archived driver and its failure remains in the training logs.
"""
import argparse
import ast
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from run_cellflow_v1 import digest, save_json

p = argparse.ArgumentParser()
p.add_argument('--run', required=True, type=Path)
p.add_argument('--baseline', required=True, type=Path)
a = p.parse_args()
run, baseline = a.run, a.baseline
assert not (run / 'test_evaluation_started.json').exists(), 'This protocol audit precedes new test evaluation.'
old = json.loads((baseline / 'config.json').read_text())
new = json.loads((run / 'config.json').read_text())
assert {k: new[k] for k in old} == old
assert set(new) - set(old) == {'training_quantiles', 'training_taus'}
assert (run / 'requirements.lock.txt').read_bytes() == (baseline / 'requirements.lock.txt').read_bytes()
assert json.loads((run / 'panel_audit.json').read_text())['panel_sha256'] == json.loads((baseline / 'panel_audit.json').read_text())['panel_sha256']

def funcs(path):
    return {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(path.read_text()).body
            if isinstance(n, ast.FunctionDef)}

functions = ['train_seed', 'trial_options', 'model_data', 'predict', 'prediction_controls', 'condition_table']
rows = []
completed, failures = [], []
for seed in new['seeds']:
    relative = Path('trials/wrf_l0p1_h0p3') / f'seed_{seed}'
    original, current = baseline / relative, run / relative
    done = current / 'training_complete.json'
    if done.exists():
        assert json.loads(done.read_text())['steps'] == 30000
        completed.append(seed)
        failure = None
    else:
        text = (run / 'logs' / f'train_seed{seed}.log').read_text()
        match = re.search(r'AssertionError: \((\d+), (nan|inf|-inf)\)', text)
        assert match, f'Unclassified unfinished seed {seed}; do not treat it as a numerical failure.'
        failure = dict(seed=seed, status='failed_nonfinite_loss', step=int(match.group(1)),
                       value=match.group(2), test_evaluated=False)
        failures.append(failure)
        save_json(current / 'training_failed.json', failure)
    fo, fn = funcs(original / 'training_source.py'), funcs(current / 'training_source.py')
    assert all(fo[k] == fn[k] for k in functions)
    before = (original / 'population_loss_source.py').read_text()
    after = (current / 'population_loss_source.py').read_text()
    assert after.replace("config.get('training_taus', spec.taus)", 'spec.taus') == before
    assert (current / 'training_cell_ids.csv').read_bytes() == (original / 'training_cell_ids.csv').read_bytes()
    assert (current / 'trial_config.json').read_bytes() == (original / 'trial_config.json').read_bytes()
    x = pd.read_csv(current / 'training_loss.csv').query('step <= 3000')
    y = pd.read_csv(original / 'training_loss.csv').query('step <= 3000')
    assert len(x) == len(y) == 3000
    assert not x.auxiliary_active.any() and not y.auxiliary_active.any()
    assert np.isfinite(x.total_loss).all()
    delta = (x.total_loss - y.total_loss).abs().to_numpy()
    assert delta[0] == 0
    first = np.flatnonzero(delta != 0)
    result = dict(passed=True, scope='Fixed configuration, data and training-function parity; not bitwise trajectory identity',
        seed=seed, training_cell_ids_identical=True, training_functions_identical=functions,
        all_original_config_settings_unchanged=True, full_environment_lock_identical=True,
        population_code_only_change='Use configured cutoff list instead of spec.taus',
        pre_auxiliary_updates=3000, first_step_loss_identical=True,
        strict_pre_auxiliary_tolerance_passed=bool(np.allclose(x.total_loss, y.total_loss, atol=5e-6, rtol=1e-5)),
        maximum_pre_auxiliary_loss_difference=float(delta.max()),
        mean_pre_auxiliary_loss_difference=float(delta.mean()),
        first_nonidentical_step=int(first[0] + 1) if len(first) else None,
        first_nonidentical_loss_difference=float(delta[first[0]]) if len(first) else 0,
        training_completed=done.exists(), failure=failure,
        model_sha256=digest(current / 'best_CellFlow.pkl') if done.exists() else None,
        interpretation='The runs share settings and seed setup but not identical numerical trajectories; the cause of divergence was not isolated.',
        verification_revision='Replaced an overstrict trajectory gate after observing training-only discrepancies; no model, hyperparameter, checkpoint, or test-driven change.')
    save_json(current / 'controlled_comparison_check.json', result)
    rows.append(result)
save_json(run / 'checks/controlled_settings_and_numerics.json', dict(passed=True, runs=rows,
    strict_original_gate_failures_retained_in_training_logs=True,
    no_models_changed=True, no_new_test_results_seen=True,
    interpretation='A matched-settings comparison, not a bitwise replay of the original training trajectories.'))
save_json(run / 'training_outcomes.json', dict(planned_seeds=new['seeds'], completed_seeds=completed,
    failures=failures, n_completed=len(completed), n_failed=len(failures),
    settings_unchanged=True, no_failed_seeds_replaced=True,
    comparison_policy='Only compare completed seeds with the same archived seeds; never call this a three-seed mean if any seed fails.'))
if not completed:
    (run / 'report').mkdir(exist_ok=True)
    (run / 'report/comparison_report.txt').write_text(
        'CellFlow 13-threshold ablation: no seed completed under unchanged settings.\n'
        'All attempted seeds are reported in training_outcomes.json. No test predictions were generated.\n'
        'The three-threshold reference completed all three seeds. Hyperparameters were not changed.\n'
        + json.dumps(failures, indent=2) + '\n')
    save_json(run / 'complete.json', dict(status='completed_with_all_training_attempts_failed',
        planned_seeds=new['seeds'], failures=failures, no_test_evaluation=True, no_github_push=True))
print(json.dumps(rows, indent=2), flush=True)
