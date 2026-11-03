"""Before test scoring, check 64-step predictions against a 128-step solve on val."""
import argparse
import json
import os
from pathlib import Path
import sys
import anndata as ad
import diffrax
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from cellflow.model import CellFlow
from run_cellflow_v1 import ROOT,REPO,PANEL,model_data,prediction_controls,condition_table
sys.path.insert(0,str(REPO))
from sciplex_eval import EvalSpec,evaluate_predictions,response_score

p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
run=a.run
assert not (run/'test_evaluation_started.json').exists()
c=json.loads((run/'config.json').read_text())
methods=json.loads((run/'selected_methods.json').read_text())['methods']
panel=ad.read_h5ad(PANEL)
train=model_data(panel[panel.obs['split'].isin(['train','control'])])
val=panel[panel.obs['split'].isin(['val','control'])].copy()
del panel
spec=EvalSpec.load(REPO/'artifacts/eval_spec_v1')
dest=run/'integration_audit';dest.mkdir(exist_ok=True)
all_checks={}
for method,trial in methods.items():
 directory=run/'trials'/trial/'seed_0'
 low=ad.read_h5ad(directory/'preds_val.h5ad')
 cf=CellFlow.load(str(directory/'best_CellFlow.pkl'))
 n=c['prediction_cells_per_condition'];source=prediction_controls(train,0,n)
 cov,metadata=condition_table(train,'val')
 raw=cf.predict(source,cov,condition_id_key='model_condition',solver=diffrax.Heun(),
                stepsize_controller=diffrax.ConstantStepSize(),dt0=1/128,max_steps=256)
 arrays=[];frames=[]
 for key,x in raw.items():
  condition,rep=key.rsplit('|',1);row=metadata[condition]
  n1=int(round(n*row['n_rep1']/row['n_cells']))
  take=n1 if rep=='rep1' else n-n1
  x=np.asarray(x[:take],dtype=np.float32);assert np.isfinite(x).all()
  arrays.append(np.maximum(x,0))
  ids=source.obs_names[source.obs.replicate.eq(rep)][:take]
  frames.append(pd.DataFrame({'condition':condition,'replicate':rep,'source_control_id':ids},
   index=[f'{key}|generated_{i}' for i in range(take)]))
 high=ad.AnnData(np.vstack(arrays),obs=pd.concat(frames),var=train.var.copy())
 assert list(high.obs_names)==list(low.obs_names)
 assert list(high.obs.source_control_id)==list(low.obs.source_control_id)
 high.write_h5ad(dest/f'{method}_val_128steps.h5ad',compression='gzip')
 lo={r['condition']:r for r in evaluate_predictions(spec,val,low,'val')}
 hi={r['condition']:r for r in evaluate_predictions(spec,val,high,'val')}
 checks=[]
 for condition in lo:
  mask=high.obs['condition'].eq(condition).to_numpy()
  rlo=response_score(low.X[mask],spec);rhi=response_score(high.X[mask],spec)
  checks.append(dict(condition=condition,primary_fraction_difference_pp=100*abs(lo[condition]['wrf_pred_q0.95']-hi[condition]['wrf_pred_q0.95']),
   response_w1_between_integrators=float(wasserstein_distance(rlo,rhi)),
   mean_abs_expression_difference=float(np.abs(low.X[mask]-high.X[mask]).mean())))
 mean_pp=float(np.mean([v['primary_fraction_difference_pp'] for v in checks]))
 max_pp=float(max(v['primary_fraction_difference_pp'] for v in checks))
 mean_w1=float(np.mean([v['response_w1_between_integrators'] for v in checks]))
 passed=mean_pp<=.25 and max_pp<=1. and mean_w1<=.01
 all_checks[method]=dict(mean_fraction_difference_pp=mean_pp,max_fraction_difference_pp=max_pp,
                         mean_response_w1_between_integrators=mean_w1,passed=passed,conditions=checks)
 print('INTEGRATION_AUDIT',method,mean_pp,max_pp,mean_w1,passed,flush=True)
 del cf,raw,low,high
result=dict(split='val',seed=0,comparison='64 vs 128 Heun steps, identical controls and conditions',
             thresholds=dict(mean_fraction_difference_pp=.25,max_fraction_difference_pp=1.,mean_response_w1=.01),
             methods=all_checks,passed=all(v['passed'] for v in all_checks.values()))
(run/'validation_integration_audit.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
assert result['passed'],'Integration sensitivity exceeded validation tolerances; test evaluation has not started.'
