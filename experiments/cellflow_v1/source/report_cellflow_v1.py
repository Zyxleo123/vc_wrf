"""Package verified CellFlow benchmark results for review and reuse."""
import argparse
import json
from pathlib import Path
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);args=p.parse_args()
run=args.run
assert (run/'complete.json').exists(),'Report requires all evaluations to finish.'
out=run/'report';out.mkdir(exist_ok=True)
config=json.loads((run/'config.json').read_text())
selection=json.loads((run/'selected_methods.json').read_text())
summaries=json.loads((run/'combined_summary.json').read_text())
metrics=pd.read_csv(run/'metric_summary.csv')
records=pd.read_csv(run/'all_per_condition_metrics.csv')
audit=json.loads((run/'panel_audit.json').read_text())
metrics.to_csv(out/'all_metric_summaries.csv',index=False)
records.to_csv(out/'all_per_condition_metrics.csv',index=False)

keys=['wrf_err_pp_q0.9','wrf_err_pp_q0.95','wrf_err_pp_q0.975','w1_r','mmd2','rmse_mean_all','rmse_mean_deg50','rmse_mean_program']
models=['cellflow','cellflow_wrf','cellflow_mmd']
names={'cellflow':'CellFlow','cellflow_wrf':'CellFlow + fraction','cellflow_mmd':'CellFlow + MMD',
 'unchanged_control':'Unchanged control','nearest_dose':'Nearest dose','adjacent_dose_mixture':'Adjacent-dose mixture',
 'replicate_reference':'Replicate reference'}
colors={'cellflow':'#2455A4','cellflow_wrf':'#DC6D2D','cellflow_mmd':'#14917A',
 'unchanged_control':'#9BA7B4','nearest_dose':'#8164B5','adjacent_dose_mixture':'#BD829B','replicate_reference':'#C2C8CE'}
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})

paired={}
rng=np.random.default_rng(42)
test=records[(records['split']=='test') & records['method'].isin(models)]
for method in models[1:]:
 paired[method]={}
 for key in keys:
  a=test[test.method.eq('cellflow')].groupby('condition')[key].mean()
  b=test[test.method.eq(method)].groupby('condition')[key].mean().reindex(a.index)
  delta=(b-a).to_numpy()
  assert len(delta)==17 and np.isfinite(delta).all()
  boots=rng.choice(delta,(2000,len(delta)),replace=True).mean(1)
  paired[method][key]={'difference_vs_cellflow':float(delta.mean()),'ci95':np.quantile(boots,[.025,.975]).tolist(),
                     'n_conditions_better':int((delta<0).sum()),'n_conditions':len(delta)}
(out/'paired_differences.json').write_text(json.dumps(paired,indent=2)+'\n')

distribution_rows=[]
for method in models:
 frames=[pd.read_csv(run/'diagnostics'/method/f'test_s{seed}_distribution_checks.csv') for seed in config['seeds']]
 frame=pd.concat(frames,ignore_index=True)
 mean=frame.groupby('condition').mean(numeric_only=True).mean()
 distribution_rows.append({'method':method,**mean.to_dict()})
distribution_summary=pd.DataFrame(distribution_rows).set_index('method')
distribution_summary.to_csv(out/'distribution_diagnostics_summary.csv')

candidate_rows=[]
for candidate in selection['candidates']:
 trial=candidate['trial']
 if candidate.get('status')=='failed':
  candidate_rows.append({k:v for k,v in candidate.items() if k not in ['selection_scores','failure']} | {'selected':False,'failure_reason':candidate['failure']['reason']})
  continue
 summary=json.loads((run/'metrics'/trial/'val_s0/summary.json').read_text())
 candidate_rows.append({**{k:v for k,v in candidate.items() if k!='selection_scores'},
  'selected':trial in selection['methods'].values(),**{k:summary[k]['mean'] for k in keys}})
pd.DataFrame(candidate_rows).to_csv(out/'validation_search.csv',index=False)

method_order=['replicate_reference','unchanged_control','nearest_dose','adjacent_dose_mixture',*models]
fig,axes=plt.subplots(2,2,figsize=(13.5,8),layout='constrained')
for ax,key,title in zip(axes.ravel(),['wrf_err_pp_q0.95','w1_r','mmd2','rmse_mean_program'],
 ['Weak-responder error at q = 0.95 (pp)','Response-score Wasserstein distance','Multigene MMD squared','Program-gene mean-expression RMSE']):
 frame=metrics[(metrics.split=='test')&(metrics.metric==key)].set_index('method').loc[method_order]
 y=np.arange(len(frame));m=frame['mean'].to_numpy()
 err=np.vstack([m-frame.ci95_low.to_numpy(),frame.ci95_high.to_numpy()-m])
 ax.barh(y,m,color=[colors[k] for k in method_order],height=.65)
 ax.errorbar(m,y,xerr=np.maximum(err,0),fmt='none',color='#303841',capsize=2,lw=1)
 ax.set(yticks=y,yticklabels=[names[k] for k in method_order],title=title,xlabel='Lower is better')
 ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
fig.suptitle('Held-out dose prediction: 17 test conditions\nModel means over 3 seeds; bars show 95% condition-bootstrap intervals',fontsize=14)
fig.savefig(out/'test_metrics.png',dpi=180);plt.close(fig)

fig,axes=plt.subplots(1,3,figsize=(15,4.2),layout='constrained')
training=[]
for method,trial in selection['methods'].items():
 for seed in config['seeds']:
  directory=run/'trials'/trial/f'seed_{seed}'
  done=json.loads((directory/'training_complete.json').read_text())
  checkpoint=json.loads((directory/'selected_checkpoint.json').read_text())
  training.append(dict(method=method,trial=trial,seed=seed,selected_step=checkpoint['step'],
                       run_wall_seconds=done['elapsed_seconds'],first_1000_loss=done['first_1000_loss'],last_1000_loss=done['last_1000_loss']))
  if method=='cellflow' and (run/'trials/cellflow_early'/f'seed_{seed}'/'training_complete.json').exists():
   training[-1]['early_checkpoint_audit_wall_seconds']=json.loads((run/'trials/cellflow_early'/f'seed_{seed}'/'training_complete.json').read_text())['elapsed_seconds']
  if seed!=0:continue
  curve=pd.read_csv(directory/'training_loss.csv')
  axes[0].plot(curve.step,curve.flow_matching_loss.rolling(500).mean(),label=names[method],color=colors[method])
  if method!='cellflow':
   ax=axes[1 if method=='cellflow_wrf' else 2]
   active=curve[curve.auxiliary_active.eq(1)]
   ax.plot(active.step,active.auxiliary_loss.rolling(100).mean(),color=colors[method])
   ax.set(title=names[method]+': auxiliary loss',xlabel='Training update',ylabel='Unweighted loss (100 active-update mean)')
axes[0].set(title='Flow-matching objective',xlabel='Training update',ylabel='Loss (500-update mean; log scale)',yscale='log');axes[0].legend(frameon=False)
fig.suptitle('Training diagnostics for the selected seed-0 settings')
fig.savefig(out/'training_curves.png',dpi=180);plt.close(fig)
pd.DataFrame(training).to_csv(out/'training_runs.csv',index=False)

split=json.loads((run/'split_v1.json').read_text())
ordered=[d for d in split['drugs_ranked_by_10uM_median_r'] if d['group']=='test']
shown=[ordered[i] for i in [0,len(ordered)//2,len(ordered)-1]]
score_frames={m:pd.read_csv(run/'diagnostics'/m/'test_s0_response_scores.csv.gz') for m in models}
spec=json.loads((run/'eval_spec_v1/spec.json').read_text())
fig,axes=plt.subplots(1,3,figsize=(15,4.5),layout='constrained')
for ax,d in zip(axes,shown):
 condition=f"A549|{d['drug']}|{d['heldout_dose_nM']:g}"
 ref=score_frames['cellflow']
 real=ref[(ref.condition==condition)&(ref.population=='real')].response_score.to_numpy()
 def ecdf(values,label,color):
  values=np.sort(values);ax.step(values,np.arange(1,len(values)+1)/len(values),where='post',label=label,color=color,lw=1.6)
 ecdf(real,'Measured cells','#222222')
 for method,frame in score_frames.items():
  vals=frame[(frame.condition==condition)&(frame.population=='generated')].response_score.to_numpy()
  ecdf(vals,names[method],colors[method])
 ax.axvline(spec['taus'][1],color='#777777',ls='--',lw=1,label='q = 0.95 threshold')
 ax.set(title=textwrap.fill(f"{d['drug']}\n{d['heldout_dose_nM']:g} nM",32),xlabel='Response score (control-SD units)',ylabel='Cumulative fraction',ylim=(0,1.02))
axes[-1].legend(frameon=False,fontsize=8)
fig.suptitle('Response distributions at held-out doses (seed 0)\nExamples fixed by strong, middle, and weak training-dose response ranks')
fig.savefig(out/'response_distributions.png',dpi=180);plt.close(fig)

lines=['CellFlow benchmark using vc_wrf v1','',
 'RESULTS COMPLETE: standard CellFlow, CellFlow + response-fraction loss, and CellFlow + MMD.',
 'Each selected method was trained with three independent seeds. Hyperparameters and checkpoints',
 'were selected using validation only, then all final test predictions were scored with the',
 'unmodified shared metric functions at commit 1d8b173f8d4120c8ba4487d7e03511b26e7a71ad.',
 '', 'Repository: https://github.com/Zyxleo123/vc_wrf/tree/1d8b173f8d4120c8ba4487d7e03511b26e7a71ad',
 '', 'DATA AND SPLIT',f"Panel: {audit['shape'][0]:,} A549 cells x {audit['shape'][1]:,} evaluation genes; 26 HDAC inhibitors, 24 hours.",
 f"Cells by split: {audit['cells_by_split']}",
 '9 validation conditions and 17 final test conditions; each drug holds out one intermediate dose.',
 'Entire treated wells are separated between train, validation, and test. All permitted vehicle',
 'controls are available to every method. Controls and predictions are matched within replicate.',
 'Evaluation gene order, normalization, response program, control statistics, PCA basis and thresholds',
 'are frozen by the repository. No test expression or test fraction is used for training or selection.',
 '', 'TRAINING',
 'The installed CellFlow OT flow-matching implementation operates directly on all 2,085 log1p(CP10K)',
 'evaluation genes. Drug identity is one-hot; dose is continuous log1p dose. Architecture and 30,000',
 'optimizer updates are matched. The auxiliary variants cost more per update; this is an update-matched',
 'comparison, not an equal-FLOP comparison. Run wall times include validation and checkpoint writes.',
 'Because the initial baseline validation curve favored its earliest saved checkpoint, a validation-only',
 'audit replayed its first 3,000 updates (same learning-rate schedule) to add checkpoints at 500, 1,000,',
 '2,000 and 3,000 updates. The better baseline checkpoint was retained separately for each seed.',
 'A pre-test fraction-loss trial at weight 1 became numerically unstable. The training response-score',
 'calculation was rewritten in an algebraically equivalent overflow-safe form, and every fraction',
 'trial was restarted using weights 0.01, 0.03 and 0.1. The failed/interrupted attempts and original',
 'configuration are retained under numerical_preflight_attempts/. No test result informed this repair.',
 f"Auxiliary losses: {config['auxiliary_batch_size']} generated cells, {config['auxiliary_ode_steps']}-step differentiable Heun rollout,",
 f"one auxiliary update per {config['auxiliary_every']} updates after update {config['auxiliary_start_step']}; the coefficient is frequency-corrected.",
 'The fraction loss uses the same smooth indicator on real and generated cells at all three frozen',
 'thresholds. MMD uses the same PCA space and bandwidths as the shared metric.',
 'Final generation: 512 cells per condition with 64 Heun steps; replicate weights follow the frozen',
 'cell counts. Negative generated log-expression is clipped to zero before all metrics.',
 '', 'SELECTED SETTINGS']
for m,t in selection['methods'].items():
 item=next(c for c in selection['candidates'] if c['trial']==t)
 steps=[r['selected_step'] for r in training if r['method']==m]
 lines.append(f"{names[m]}: {t}; weight={item['weight']:g}; smoothing h={item['h']:g}; selected steps by seed={steps}.")
failed=[v for v in selection['candidates'] if v.get('status')=='failed']
if failed:
 lines+=['','FAILED VALIDATION CANDIDATES']
 for v in failed:
  lines.append(f"{v['trial']}: {v['failure']['reason']} at update {v['step']}; preserved in trials/ and excluded from selection. No test result was used.")
lines+=['','TEST METRICS','Lower is better for all metrics. WRF errors are percentage points.','Values are equal-weight means over 17 conditions after averaging the three training seeds.','95% CIs resample conditions; they do not imply 51 independent biological experiments.','']
for key in keys:
 lines.append(key)
 for method in models:
  v=summaries[f'{method}/test'][key]
  row=metrics[(metrics.method==method)&(metrics.split=='test')&(metrics.metric==key)].iloc[0]
  lines.append(f"  {names[method]:22s} {v['mean']:.6f}  95% CI [{v['ci95'][0]:.6f}, {v['ci95'][1]:.6f}]; seed SD {row.training_seed_sd:.6f}")
lines+=['','PAIRED PRIMARY-METRIC DIFFERENCES (method minus standard CellFlow; negative is better)']
for method in models[1:]:
 v=paired[method]['wrf_err_pp_q0.95']
 lines.append(f"{names[method]}: {v['difference_vs_cellflow']:.4f} pp [{v['ci95'][0]:.4f}, {v['ci95'][1]:.4f}]; improves {v['n_conditions_better']}/17 conditions.")
lines+=['','ADDITIONAL DISTRIBUTION CHECKS',
 'Descriptive averages over test conditions and seeds; these were not selection criteria.',
 'Variance ratio is generated / measured total gene variance (1 means equal). Zero fractions',
 'are the fraction of gene-expression entries equal to zero after stated postprocessing.']
for method in models:
 row=distribution_summary.loc[method]
 lines.append(f"{names[method]}: non-program mean RMSE={row.rmse_mean_nonprogram:.6f}; variance ratio={row.total_variance_ratio:.4f}; zero fraction measured={row.mean_gene_zero_fraction_real:.4f}, predicted={row.mean_gene_zero_fraction_pred:.4f}.")
lines+=['','INTERPRETATION LIMITS',
 'The repository warns that the current q=0.95 weak-responder endpoint is weakly sensitive to drug',
 'effects and q=0.975 is degenerate because many cells have zero detected program expression.',
 'Thresholds were kept unchanged. WRF must therefore be read together with W1, MMD, gene-mean',
 'errors, per-well results, and response distributions. A matched fraction alone does not establish',
 'a correct response distribution. Replicate disagreement is a reference, not a strict lower bound.',
 'Only known-drug dose interpolation in A549 is tested. There are two replicate wells per condition.',
 'This is a CellFlow-based benchmark implementation, not a reproduction of every setting in the',
 'published CellFlow paper. The three primary variants are complete; other architectures such as',
 'scVIDR are separate project comparisons.',
 '', 'SAVED ARTIFACTS',str(run),
 'config.json, source_manifest.json, requirements.lock.txt, panel_audit.json, frozen spec/split,',
 'all validation candidate summaries, selected_methods.json, models and predictions under trials/,',
 'per-condition metrics, per-well metrics, per-cell response scores, distribution checks and figures.',
 'Each selected trial has best_CellFlow.pkl, preds_val.h5ad, preds_test.h5ad, training_loss.csv,',
 'training_cell_ids.csv, validation_history.json and selected_checkpoint.json.',
 'Seed 0 also has preds_train.h5ad and metrics for all 78 seen treatment conditions; these',
 'are descriptive in-sample diagnostics and were never used for checkpoint or hyperparameter choice.',
 'No test result was used to revise the selected models. Test model/config hashes are frozen in',
 'test_evaluation_started.json. The shared data remain under',
 '/ocean/projects/cis260266p/shared/response-fraction/data.', '']
lines+=['ADDITIONAL UNCERTAINTY AND NUMERICAL CHECKS',
 'A run seed changes both model training and the sampled source controls; the reported seed SD',
 'therefore includes both sources of variation. It is not an estimate from independent biological',
 'replicates. Per-well metrics and the replicate reference are saved separately.',
 'Interrupted runs are retained under interrupted_allocation_50003709/ and interrupted_allocation_50004068/.',
 'The remaining experiment uses an active allocation driver rather than an idle login shell.', '']
if (run/'reference_compatibility_record.json').exists():
 lines+=['Pandas 3.1 makes the official replicate helper\'s NumPy mask read-only. A local adapter',
         'copies that mask before combining selectors, then calls the unchanged official metric',
         'functions in both directions and averages them exactly as the original helper does.',
         'The pinned metric repository, model choices, generated predictions, and metric definitions',
         'remain unchanged. See reference_compatibility_record.json for the equivalence check.', '']
if (run/'validation_integration_audit.json').exists():
 integration=json.loads((run/'validation_integration_audit.json').read_text())
 lines.append('Validation numerical integration check: 64 vs 128 Heun steps, with identical source cells.')
 for method,check in integration['methods'].items():
  lines.append(f"  {names[method]}: mean fraction difference {check['mean_fraction_difference_pp']:.5f} pp, max {check['max_fraction_difference_pp']:.5f} pp; mean response W1 {check['mean_response_w1_between_integrators']:.6f}.")
 lines.append('')
(out/'cellflow-v1-report.txt').write_text('\n'.join(lines))
from verify_cellflow_artifacts import verify
verify(run)
print('REPORT_COMPLETE',str(out),flush=True)
