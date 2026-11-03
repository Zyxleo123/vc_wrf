"""Report all seeds of both repaired WRF arms and preserved baseline references."""
import argparse
import json
from pathlib import Path
import hashlib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
root=a.root;out=root/'report';out.mkdir(exist_ok=True)
run=root/'run13'
assert json.loads((run/'evaluation_complete.json').read_text())['new_models']==6
metrics=pd.read_csv(run/'report/metric_summary.csv')
records=pd.read_csv(run/'report/all_per_condition_metrics.csv')
grid=pd.read_csv(run/'report/threshold_grid_per_condition.csv')
paired=pd.read_csv(run/'report/paired_differences.csv')
for name,frame in [('metric_summary.csv',metrics),('all_per_condition_metrics.csv',records),
                   ('threshold_grid_per_condition.csv',grid),('paired_differences.csv',paired)]:
    frame.to_csv(out/name,index=False)
names={'cellflow':'Standard CellFlow (archived)','cellflow_wrf':'CellFlow + WRF, 3 thresholds (repaired)',
       'cellflow_wrf13':'CellFlow + WRF, 13 thresholds (repaired)','cellflow_mmd':'CellFlow + MMD (archived)',
       'unchanged_control':'Vehicle cells','nearest_dose':'Nearest dose','adjacent_dose_mixture':'Adjacent-dose mixture',
       'replicate_reference':'Replicate disagreement'}
models=['cellflow','cellflow_wrf','cellflow_wrf13','cellflow_mmd']
keys=['wrf_err_pp_q0.9','wrf_err_pp_q0.95','wrf_err_pp_q0.975','wrf_err_pp_grid13_mean',
      'w1_r','mmd2','rmse_mean_all','rmse_mean_deg50','rmse_mean_program']
test=metrics[metrics.split=='test'].set_index(['method','metric'])
training=[]
frozen=json.loads((root/'all_models_frozen_before_test.json').read_text())
for count in [3,13]:
    for seed in [0,1,2]:
        d=root/f'run{count}'/'trials/wrf_l0p1_h0p3'/f'seed_{seed}'
        check=json.loads((d/'numerical_training_check.json').read_text())
        done=json.loads((d/'training_complete.json').read_text())
        selected=json.loads((d/'selected_checkpoint.json').read_text())
        assert check['passed'] and done['steps']==30000
        assert hashlib.sha256((d/'best_CellFlow.pkl').read_bytes()).hexdigest()==frozen[str(count)]['models'][str(seed)]['model_sha256']
        training.append(dict(thresholds=count,seed=seed,selected_step=selected['step'],
            seconds=done['elapsed_seconds'],finite_state_checks=check['finite_parameter_and_optimizer_checks']))
pd.DataFrame(training).to_csv(out/'training_summary.csv',index=False)
for _,r in metrics.iterrows():
    part=records[(records.method==r.method)&(records.split==r.split)]
    means=part.groupby('seed')[r.metric].mean()
    np.testing.assert_allclose(means.mean(),r['mean'],rtol=1e-10,atol=1e-10)
    if r.method in models:
        assert len(means)==3
        np.testing.assert_allclose(means.std(ddof=1),r.seed_sd,rtol=1e-10,atol=1e-10)
lines=['CellFlow WRF: full three-seed comparison after numerical repair','',
       'Both WRF arms completed all three seeds and 30,000 updates per seed.',
       'The original 13-threshold attempt remains preserved (two seeds failed).',
       'The repair uses float64 only in the differentiable 8-step auxiliary trajectory and response score.',
       'The ordinary flow-matching calculation, parameters and optimizer retain their original precision.',
       'No skipped updates, replaced NaNs, clipped trajectories or changed hyperparameters were used.',
       'Weight=.1, h=.3; architecture, learning-rate schedule, seeds, data, split, EMA and clipping unchanged.',
       'Both threshold arms were rerun so they use one numerical implementation.',
       'Checkpoint rule: validation q0.95 error then W1, evaluated at 5k/10k/20k/30k.',
       'All six selected models were frozen before new test scoring.',
       'Final predictions use 512 cells/condition, replicate-matched controls and 64 Heun steps.','',
       'TEST: 17 conditions, equal weighting. Mean +/- sample SD over 3 training seeds.',
       'All nine metric columns use per-seed errors before averaging. WRF units: percentage points.',
       'Archived CellFlow and MMD baselines retain the original implementation.','']
for key in keys:
    lines.append(key)
    for m in models:
        r=test.loc[m,key]
        lines.append(f"  {names[m]}: {r['mean']:.6f} +/- {r.seed_sd:.6f}; condition95%CI[{r.ci95_low:.6f},{r.ci95_high:.6f}]")
lines+=['','PAIRED DIFFERENCE: 13 minus 3 thresholds; negative favors 13.']
for _,r in paired[paired.reference=='cellflow_wrf'].iterrows():
    lines.append(f"  {r.metric}: {r.difference:+.6f}; 95% CI [{r.ci95_low:+.6f},{r.ci95_high:+.6f}]; improved {r.conditions_improved}/17 conditions")
lines+=['','LIMITATIONS',
        'Exploratory repair and rerun after prior test results were seen. The repair was chosen from',
        'captured training failures and numerical checks, not new test performance. No hyperparameter search.',
        'Seed variation includes training and source-control sampling. Condition bootstrap intervals',
        'are not confidence intervals across independent biological experiments.',
        'The q0.95 endpoint is weakly sensitive and q0.975 lies at the zero-program-count boundary.',
        'Replicate disagreement is a reference, not a strict lower bound.',
        'The existing chemCPA grid column averages fractions across seeds before absolute error;',
        'harmonize that convention before making a final cross-model grid comparison.',
        'Higher precision prevents arithmetic overflow; it does not guarantee that an 8-step auxiliary',
        'trajectory is an accurate ODE solution during training. Validation-selected predictions were',
        'checked at 64 versus 128 steps on seed 0 before test evaluation.','',
        'TRAINING',json.dumps(training,indent=2)]
(out/'comparison_report.txt').write_text('\n'.join(lines)+'\n')
mainkeys=['wrf_err_pp_q0.95','wrf_err_pp_grid13_mean','w1_r','mmd2','rmse_mean_deg50','rmse_mean_program']
compact=[]
for m in [*models,'unchanged_control','nearest_dose','adjacent_dose_mixture','replicate_reference']:
    compact.append({'method':names[m],**{k:test.loc[m,k]['mean'] for k in mainkeys},
                    'primary_seed_sd':test.loc[m,'wrf_err_pp_q0.95'].seed_sd,
                    'n_seeds':test.loc[m,'wrf_err_pp_q0.95'].n_seeds})
pd.DataFrame(compact).to_csv(out/'comparison_table.csv',index=False)

plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})
colors=['#687386','#2673b8','#cf6c2c','#2b917a']
labels=['Standard CellFlow','WRF: 3 thresholds','WRF: 13 thresholds','CellFlow + MMD']
fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
for ax,key,title in zip(axes.flat,mainkeys[:4],['Primary WRF error (pp)','13-threshold mean error (pp)','Response-score W1','Multigene MMD²']):
    values=test.loc[[(m,key) for m in models]]
    ax.barh(range(4),values['mean'],color=colors)
    ax.errorbar(values['mean'],range(4),xerr=values.seed_sd,fmt='none',color='#333333',capsize=3)
    ax.set(yticks=range(4),yticklabels=labels,title=title,xlabel='Lower is better');ax.invert_yaxis()
    ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
fig.suptitle('CellFlow: 3 versus 13 thresholds after numerical repair\n17 test conditions; 3 seeds; error bars = seed SD')
fig.savefig(out/'comparison.png',dpi=180);plt.close(fig)
fig,ax=plt.subplots(figsize=(9,5),layout='constrained')
for m,label,color in zip(models,labels,colors):
    part=grid[(grid.split=='test')&(grid.method==m)].groupby('q').err_pp.mean()
    ax.plot(part.index,part,marker='o',markersize=4,color=color,label=label)
ax.axvline(.95,color='#888888',linestyle='--',linewidth=1)
ax.set(xlabel='Control response-score quantile q',ylabel='WRF absolute error (percentage points)',title='Three-seed threshold sensitivity after numerical repair')
ax.legend(frameon=False);ax.grid(alpha=.2);fig.savefig(out/'threshold_sensitivity.png',dpi=180);plt.close(fig)
status=dict(status='complete',repaired_models=6,seeds=[0,1,2],training_failures=0,
    full_three_seed_comparison=True,summaries_verified=True,test_model_hashes_unchanged=True,
    prior_failed_attempts_preserved=True,no_github_push=True)
(root/'complete.json').write_text(json.dumps(status,indent=2)+'\n')
print(json.dumps(status),flush=True)
