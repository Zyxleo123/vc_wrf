"""Verify published CellFlow provenance and metrics using only the standard library."""
import csv
from collections import defaultdict
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from statistics import fmean, stdev

REPO=Path(__file__).resolve().parents[2]
MODELS=['cellflow','cellflow_wrf','cellflow_wrf13','cellflow_mmd']
METRICS=['wrf_err_pp_q0.9','wrf_err_pp_q0.95','wrf_err_pp_q0.975','w1_r','mmd2',
         'rmse_mean_all','rmse_mean_deg50','rmse_mean_program','wrf_err_pp_grid13_mean']

def read(path):return json.loads(path.read_text())
def rows(path):
    with path.open(newline='') as f:return list(csv.DictReader(f))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def close(a,b,label):
    assert math.isfinite(a) and math.isclose(a,b,rel_tol=1e-9,abs_tol=1e-9),(label,a,b)

def check_report(directory,seeds):
    data=rows(directory/'report/all_per_condition_metrics.csv')
    summaries=rows(directory/'report/metric_summary.csv')
    grid=rows(directory/'report/threshold_grid_per_condition.csv')
    grid_errors=defaultdict(list)
    for r in grid:
        close(float(r['err_pp']),100*abs(float(r['wrf_pred'])-float(r['wrf_real'])),'grid WRF')
        grid_errors[r['method'],r['seed'],r['split'],r['condition']].append(r)
    split_manifest=read(REPO/'configs/split_v1.json')
    for split,n_conditions in [('val',9),('test',17)]:
        expected={r['condition'] for r in split_manifest['conditions'] if r['split']==split}
        assert len(expected)==n_conditions
        for method in MODELS:
            part=[r for r in data if r['method']==method and r['split']==split]
            assert {int(r['seed']) for r in part}==set(seeds)
            assert len(part)==n_conditions*len(seeds)
            for seed in seeds:
                selected=[r for r in part if int(r['seed'])==seed]
                assert {r['condition'] for r in selected}==expected
                for r in selected:
                    assert int(float(r['n_pred']))==512
                    assert all(math.isfinite(float(r[m])) for m in METRICS)
                    for q in ['0.9','0.95','0.975']:
                        close(float(r['wrf_err_pp_q'+q]),100*abs(float(r['wrf_pred_q'+q])-float(r['wrf_real_q'+q])),q)
                    g=grid_errors[method,str(seed),split,r['condition']]
                    assert len(g)==13 and len({float(x['q']) for x in g})==13
                    close(float(r['wrf_err_pp_grid13_mean']),fmean(float(x['err_pp']) for x in g),'grid mean')
            for metric in METRICS:
                means=[fmean(float(r[metric]) for r in part if int(r['seed'])==s) for s in seeds]
                summary=[r for r in summaries if r['method']==method and r['split']==split and r['metric']==metric]
                assert len(summary)==1
                s=summary[0];close(float(s['mean']),fmean(means),(method,metric))
                assert int(s['n_seeds'])==len(seeds) and int(s['n_conditions'])==n_conditions
                if len(seeds)>1:close(float(s['seed_sd']),stdev(means),'sample seed SD')
                else:assert not s['seed_sd'] or math.isnan(float(s['seed_sd']))
    return len(MODELS)*len(seeds)*17

def verify():
    spec=importlib.util.spec_from_file_location('verify_v1',REPO/'experiments/cellflow_v1/verify_published_results.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    original=module.verify(REPO)
    initial=REPO/'docs/cellflow/q13_initial_20261030'
    outcomes=read(initial/'training_outcomes.json')
    assert outcomes['completed_seeds']==[0]
    assert {(f['seed'],f['step']) for f in outcomes['failures']}=={(1,3550),(2,3150)}
    hashes=read(initial/'artifact_sha256.json')
    for r in read(initial/'published_source_provenance.json'):
        assert sha(REPO/r['path'])==r['sha256']==hashes[r['original_path']],r['path']
    initial_rows=check_report(initial,[0])

    repaired=REPO/'docs/cellflow/q13_repair_20261030'
    for path,expected in read(repaired/'local_package_manifest.json').items():
        assert sha(REPO/path)==expected,path
    configs=[read(repaired/f'run{q}/config.json') for q in [3,13]]
    assert [k for k in configs[0] if configs[0][k]!=configs[1][k]]==['training_quantiles','training_taus']
    grid=read(REPO/'configs/eval_grid13_v1.json')
    assert grid['quantiles']==configs[1]['training_quantiles'] and grid['taus']==configs[1]['training_taus']
    for count in [3,13]:
        assert read(repaired/f'run{count}/validation_integration_audit.json')['passed']
        for seed in [0,1,2]:
            d=repaired/f'run{count}/seed_{seed}'
            assert read(d/'training_complete.json')['steps']==30000
            assert read(d/'numerical_training_check.json')['finite_parameter_and_optimizer_checks']==5400
            selected=read(d/'selected_checkpoint.json');assert selected['selected_on']=='validation only'
    assert read(REPO/'experiments/cellflow_q13_repair_20261030/diagnosis/precision_repair_check.json')['passed']
    assert read(REPO/'experiments/cellflow_q13_repair_20261030/diagnosis/gradient_repair_check.json')['passed']
    repaired_rows=check_report(repaired,[0,1,2])
    complete=read(repaired/'complete.json');assert complete['repaired_models']==6 and complete['training_failures']==0
    manifest=REPO/'docs/cellflow/publication_20261102.json'
    publication_files=0
    if manifest.exists():
        for r in read(manifest)['files']:
            assert sha(REPO/r['path'])==r['sha256'],r['path'];publication_files+=1
    return dict(passed=True,original_three_seed_test_rows=original['selected_test_rows'],
        initial_seed0_comparison_test_rows=initial_rows,precision_rerun_comparison_test_rows=repaired_rows,
        new_completed_training_runs=6,initial_failed_seeds_preserved=[1,2],
        shared_evaluator_unchanged=True,publication_files_checked=publication_files,
        scope='Saved metrics, source hashes and recorded numerical checks; no new training or prediction rescoring.')

if __name__=='__main__':print(json.dumps(verify(),indent=2))
