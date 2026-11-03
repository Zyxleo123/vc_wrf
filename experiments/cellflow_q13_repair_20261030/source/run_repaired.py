"""Matched three-seed reruns after a validated precision-only numerical repair."""
import argparse
import json
import os
from pathlib import Path
import shutil

import anndata as ad
import jax
import jax.numpy as jnp
import numpy as np

import run_cellflow_v1 as base
import cellflow_population_loss as population
from sciplex_eval import EvalSpec
from score_comparison import comparison, TRIAL, METHOD

BASELINE=Path('/ocean/projects/cis260266p/ywang179/response-fraction/results/cellflow_vc_wrf_v1_50003709')
PREVIOUS=Path('/ocean/projects/cis260266p/ywang179/response-fraction/experiments/cellflow_q13_20261030/run')
REPAIR=Path('/ocean/projects/cis260266p/ywang179/response-fraction/experiments/cellflow_q13_repair_20261030')


def initialize(root):
    assert json.loads((REPAIR/'precision_repair_check.json').read_text())['passed']
    previous=json.loads((PREVIOUS/'config.json').read_text())
    spec=EvalSpec.load(base.REPO/'artifacts/eval_spec_v1')
    (root/'logs').mkdir(exist_ok=True)
    shutil.copytree(base.REPO/'artifacts/eval_spec_v1',root/'eval_spec_v1')
    shutil.copyfile(base.REPO/'configs/split_v1.json',root/'split_v1.json')
    provenance=root/'diagnosis';provenance.mkdir()
    for name in ['precision_repair_check.json','diagnosis_seed1/failure_diagnosis.json','diagnosis_seed2/no_failure_reproduced.json']:
        shutil.copyfile(REPAIR/name,provenance/Path(name).name)
    for name in ['diagnose_failure.py','check_precision_repair.py','cellflow_population_loss.py','cellflow_population_loss_fp64.py']:
        shutil.copyfile(REPAIR/'src'/name,provenance/name)
    for count in [3,13]:
        run=root/f'run{count}'
        run.mkdir(exist_ok=False)
        config=previous.copy()
        if count==3:
            config['training_quantiles']=list(spec.quantiles)
            config['training_taus']=list(spec.taus)
        assert {k:config[k] for k in previous if k not in ['training_quantiles','training_taus']} == {k:previous[k] for k in previous if k not in ['training_quantiles','training_taus']}
        base.save_json(run/'config.json',config)
        method=METHOD if count==13 else 'cellflow_wrf'
        base.save_json(run/'selected_methods.json',dict(methods={method:TRIAL},selected_on='Previously fixed lambda=.1, h=.3; validation-only checkpoint rule.'))
        base.audit_panel(run)
        assert json.loads((run/'panel_audit.json').read_text())['panel_sha256']==json.loads((BASELINE/'panel_audit.json').read_text())['panel_sha256']
        shutil.copytree(root/'src',run/'source',ignore=shutil.ignore_patterns('vc_wrf_1d8b173','__pycache__'))
        shutil.copyfile(REPAIR/'precision_repair_check.json',run/'precision_repair_check.json')
        shutil.copyfile(PREVIOUS/'requirements.lock.txt',run/'requirements.lock.txt')
        base.save_json(run/'source_manifest.json',{str(p.relative_to(run)):base.digest(p) for p in run.rglob('*') if p.is_file()})
    protocol=dict(planned_seeds=[0,1,2],arms=[3,13],weight=.1,h=.3,steps=30000,
        baseline_run=str(BASELINE),previous_failed_experiment=str(PREVIOUS),
        repair_validation=str(REPAIR/'precision_repair_check.json'),
        numerical_repair='Float64 auxiliary Heun trajectory and response score; float32 ordinary flow loss, trainable parameters and optimizer retained.',
        unchanged='Architecture, data, split, random seeds, learning rate, optimizer, gradient clipping, EMA, 8 auxiliary Heun steps, 64 prediction steps, loss weight, h, checkpoint rule and evaluation definitions.',
        no_clipped_trajectories=True,no_skipped_batches=True,no_nan_replacement=True,
        paired_design='Rerun both threshold sets for all three seeds using the same repaired implementation.',
        test_policy='Freeze all six selected models before any new test scoring; no test-based tuning.',
        exploratory=True,reason='Numerical repair requested after the original failures and exploratory test results.',
        no_github_push=True)
    base.save_json(root/'repair_protocol.json',protocol)
    print('REPAIRED_RUNS_INITIALIZED',json.dumps(protocol),flush=True)


def train(root,count,seed):
    run=root/f'run{count}'
    directory=run/'trials'/TRIAL/f'seed_{seed}'
    original=population.make_population_step
    checks=0
    @jax.jit
    def all_finite(tree):
        return jnp.all(jnp.stack([jnp.all(jnp.isfinite(x)) for x in jax.tree.leaves(tree)]))
    def instrument(solver,*args,**kwargs):
        update=original(solver,*args,**kwargs)
        def step(rng,batch):
            nonlocal checks
            values=update(rng,batch)
            if bool(values[3]):
                assert bool(all_finite((solver.vf_state.params,solver.vf_state.opt_state))), 'Nonfinite model/optimizer state after auxiliary update'
                checks+=1
            return values
        return step
    population.make_population_step=instrument
    try:
        base.train_seed(run,seed,TRIAL)
    except Exception as exc:
        base.save_json(directory/'training_failed.json',dict(seed=seed,thresholds=count,error=repr(exc),test_evaluated=False))
        raise
    finally:
        population.make_population_step=original
    ids=directory/'training_cell_ids.csv'
    old=BASELINE/'trials'/TRIAL/f'seed_{seed}'/'training_cell_ids.csv'
    assert ids.read_bytes()==old.read_bytes()
    assert checks==(30000-3000)//5
    base.save_json(directory/'numerical_training_check.json',dict(passed=True,seed=seed,thresholds=count,
        finite_parameter_and_optimizer_checks=checks,training_ids_identical_to_original=True,
        updates=30000,configuration_unchanged_except_thresholds=True))
    print('REPAIRED_TRAINING_VERIFIED',count,seed,checks,flush=True)


def evaluate(root):
    from cellflow.model import CellFlow
    frozen={}
    for count in [3,13]:
        run=root/f'run{count}'
        assert json.loads((run/'validation_integration_audit.json').read_text())['passed']
        config=json.loads((run/'config.json').read_text())
        entries={}
        for seed in [0,1,2]:
            directory=run/'trials'/TRIAL/f'seed_{seed}'
            assert json.loads((directory/'numerical_training_check.json').read_text())['passed']
            assert json.loads((directory/'training_complete.json').read_text())['steps']==30000
            entries[str(seed)]=dict(checkpoint=json.loads((directory/'selected_checkpoint.json').read_text()),
                model_sha256=base.digest(directory/'best_CellFlow.pkl'))
        frozen[str(count)]=dict(config_sha256=base.digest(run/'config.json'),models=entries)
    path=root/'all_models_frozen_before_test.json'
    if path.exists(): assert json.loads(path.read_text())==frozen
    else: base.save_json(path,frozen)
    panel=ad.read_h5ad(base.PANEL)
    train=base.model_data(panel[panel.obs['split'].isin(['train','control'])])
    for count in [3,13]:
        run=root/f'run{count}'
        base.save_json(run/'test_evaluation_started.json',frozen[str(count)])
        config=json.loads((run/'config.json').read_text())
        for seed in [0,1,2]:
            directory=run/'trials'/TRIAL/f'seed_{seed}'
            if not (directory/'preds_test.h5ad').exists():
                model=CellFlow.load(str(directory/'best_CellFlow.pkl'))
                pred=base.predict(model,train,config,'test',seed)
                pred.write_h5ad(directory/'preds_test.tmp.h5ad',compression='gzip')
                (directory/'preds_test.tmp.h5ad').rename(directory/'preds_test.h5ad')
                del model,pred
            assert base.digest(directory/'best_CellFlow.pkl')==frozen[str(count)]['models'][str(seed)]['model_sha256']
            print('GENERATED_REPAIRED_TEST',count,seed,flush=True)
    run=root/'run13'
    base.save_json(run/'training_outcomes.json',dict(planned_seeds=[0,1,2],completed_seeds=[0,1,2],failures=[],
        settings_unchanged=True,implementation='validated mixed-precision numerical repair',matched_three_threshold_rerun=True))
    spec=EvalSpec.load(base.REPO/'artifacts/eval_spec_v1')
    comparison(run,BASELINE,panel,spec,json.loads((run/'config.json').read_text()))
    print('FULL_THREE_SEED_COMPARISON_SCORED',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('stage',choices=['initialize','train','evaluate'])
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--count',type=int,choices=[3,13])
    p.add_argument('--seed',type=int,choices=[0,1,2])
    args=p.parse_args()
    if args.stage=='initialize':initialize(args.root)
    elif args.stage=='train':train(args.root,args.count,args.seed)
    else:evaluate(args.root)
