"""Train CellFlow and use the unmodified vc_wrf v1 evaluation package.

An explicit loop around the installed CellFlow sampler and OT solver preserves
seed-specific random streams across validation checkpoints. The base variant uses
the standard solver; auxiliary variants add generated-population penalties.
"""
import argparse
import functools
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys
import time

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

ROOT = Path(os.environ['RESPONSE_FRACTION_ROOT'])
DATA = Path(os.environ['RESPONSE_FRACTION_DATA'])
REPO = ROOT / 'src/vc_wrf_1d8b173'
PANEL = DATA / 'processed/a549_hdaci_v1.h5ad'
sys.path.insert(0, str(REPO))
from sciplex_eval import EvalSpec, evaluate_condition, evaluate_predictions, replicate_reference, response_score, summarize
from sciplex_eval.spec import program_activity

def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')

def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while block := f.read(8*1024*1024):
            h.update(block)
    return h.hexdigest()

def dense(x):
    return x.toarray() if sparse.issparse(x) else np.asarray(x)

def audit_panel(run):
    """Verify the rebuilt data against frozen metadata and control statistics."""
    a = ad.read_h5ad(PANEL)
    spec = EvalSpec.load(REPO/'artifacts/eval_spec_v1')
    split = json.loads((REPO/'configs/split_v1.json').read_text())
    assert a.shape == (24786, 2085), a.shape
    assert a.obs_names.is_unique and a.var_names.is_unique
    assert list(a.var_names) == spec.gene_ids
    assert set(a.obs['split']) == {'control', 'train', 'val', 'test'}
    ctrl = a.obs['split'].eq('control').to_numpy()
    assert ctrl.sum() == split['controls']['n_cells']
    for c in split['conditions']:
        rows = a.obs[a.obs['condition'].eq(c['condition'])]
        assert len(rows) == c['n_cells']
        assert rows['split'].eq(c['split']).all()
        assert rows['replicate'].eq('rep1').sum() == c['n_rep1']
        assert rows['replicate'].eq('rep2').sum() == c['n_rep2']
    act = program_activity(a.X[ctrl], spec)
    np.testing.assert_allclose([act.mean(), act.std(ddof=1)], [spec.mu0,spec.s0], rtol=2e-6)
    np.testing.assert_allclose(np.asarray(a.X[ctrl].mean(0)).ravel(), spec.control_mean, atol=2e-6)
    np.testing.assert_allclose(np.quantile(response_score(a.X[ctrl],spec), spec.quantiles), spec.taus, atol=2e-6)
    wells = a.obs['culture_plate'].astype(str) + '|' + a.obs['well_oligo'].astype(str)
    sets = {s:set(wells[a.obs['split'].eq(s)]) for s in ['train','val','test']}
    for x,y in [('train','val'),('train','test'),('val','test')]:
        assert not sets[x].intersection(sets[y]), (x,y,'well overlap')
    assert np.isfinite(a.X.data).all() and (a.X.data >= 0).all()
    result = dict(shape=list(a.shape), cells_by_split={k:int(v) for k,v in a.obs['split'].value_counts().items()},
                  conditions_by_split={k:int(v) for k,v in a.obs.groupby('split',observed=True)['condition'].nunique().items()},
                  panel_sha256=digest(PANEL),frozen_spec_matches=True,treated_well_overlap=False)
    save_json(run/'panel_audit.json',result)
    print('PANEL_AUDIT',json.dumps(result),flush=True)

def model_data(a):
    a = a.copy()
    a.obs['is_control'] = a.obs['split'].eq('control')
    a.obs['log_dose'] = (np.log1p(a.obs['dose'].astype(float))/np.log1p(10000)).astype(np.float32)
    a.obs['model_condition'] = a.obs['condition'].astype(str) + '|' + a.obs['replicate'].astype(str)
    a.X = dense(a.X).astype(np.float32)
    drugs = sorted(a.obs.loc[~a.obs['is_control'], 'drug'].astype(str).unique())
    a.uns['drug_embeddings'] = {d:np.eye(len(drugs),dtype=np.float32)[i] for i,d in enumerate(drugs)}
    a.uns['drug_embeddings']['Vehicle'] = np.zeros(len(drugs),dtype=np.float32)
    return a

def condition_table(train, split_name):
    """Prediction targets from frozen labels, without accessing held-out expression."""
    split = json.loads((REPO/'configs/split_v1.json').read_text())
    selected = [c for c in split['conditions'] if c['split']==split_name]
    rows=[]
    for c in selected:
        for rep in ['rep1','rep2']:
            row=train.obs[(train.obs.drug==c['drug']) & (train.obs.replicate==rep)].iloc[0].copy()
            row['condition']=c['condition']
            row['model_condition']=c['condition']+'|'+rep
            row['dose']=c['dose_nM']
            row['log_dose']=np.float32(np.log1p(c['dose_nM'])/np.log1p(10000))
            row['split']=split_name
            rows.append(row)
    return pd.DataFrame(rows).reset_index(drop=True), {c['condition']:c for c in selected}

def prediction_controls(train, seed, n):
    rng = np.random.default_rng(10000+seed)
    ids=[]
    for rep in ['rep1','rep2']:
        pool=train.obs.index[train.obs['is_control'] & train.obs['replicate'].eq(rep)].to_numpy()
        assert len(pool)>=n
        ids.extend(rng.choice(pool,n,replace=False))
    return train[ids].copy()

def predict(cf, train, config, split_name, seed):
    import diffrax
    n=config['prediction_cells_per_condition']
    source=prediction_controls(train,seed,n)
    cov,metadata=condition_table(train,split_name)
    out=cf.predict(source,cov,condition_id_key='model_condition',solver=diffrax.Heun(),
                   stepsize_controller=diffrax.ConstantStepSize(),dt0=1/64,max_steps=128)
    arrays=[]; frames=[]; negative=0; elements=0
    for key,raw in out.items():
        raw=np.asarray(raw,dtype=np.float32)
        assert raw.shape==(n,train.n_vars) and np.isfinite(raw).all(),(key,raw.shape)
        condition,rep=key.rsplit('|',1)
        c=metadata[condition]
        n_rep1=int(round(n*c['n_rep1']/c['n_cells']))
        take=n_rep1 if rep=='rep1' else n-n_rep1
        raw=raw[:take]
        negative+=int((raw<0).sum()); elements+=int(raw.size)
        arrays.append(np.maximum(raw,0))
        source_ids=source.obs_names[source.obs.replicate.eq(rep)][:take]
        frames.append(pd.DataFrame({'condition':condition,'replicate':rep,'source_control_id':source_ids},
                      index=[f'{key}|generated_{i}' for i in range(take)]))
    pred=ad.AnnData(np.vstack(arrays),obs=pd.concat(frames),var=train.var.copy())
    pred.uns['normalization']='log1p(CP10K), human-gene library size'
    pred.uns['postprocessing']=config['prediction_postprocessing']
    pred.uns['negative_fraction_before_clipping']=negative/elements
    for c,group in pred.obs.groupby('condition',observed=True):
        assert len(group)==n and n>=metadata[c]['n_cells']
    return pred

def write_metrics(run, method, split_name, seed, records, checkpoint=None):
    destination=run/'metrics'/method/f'{split_name}_s{seed}'
    destination.mkdir(parents=True,exist_ok=True)
    meta=dict(method=method,split=split_name,seed=seed,spec_version='v1',evaluation_commit='1d8b173f8d4120c8ba4487d7e03511b26e7a71ad')
    if checkpoint is not None:
        meta['training_step']=checkpoint
    with open(destination/'per_condition.jsonl','w') as f:
        for rec in records:
            f.write(json.dumps({**meta,**rec},allow_nan=False)+'\n')
    summary={**meta,**summarize(records)}
    save_json(destination/'summary.json',summary)
    return summary

def replicate_reference_compatible(spec, panel, split_name, seed=0):
    """Official bidirectional reference with writable masks for pandas 3.1.

    The pinned helper mutates Series.values in place, which is now read-only.
    Only the row selectors differ; both directions use the unmodified official
    evaluate_condition and the same arithmetic mean over every returned field.
    """
    records=[]
    conditions=sorted(panel.obs.loc[panel.obs['split'].eq(split_name),'condition'].unique())
    for condition in conditions:
        populations=[]
        for replicate in ['rep1','rep2']:
            mask=panel.obs['condition'].eq(condition).to_numpy(dtype=bool,copy=True)
            mask &= panel.obs['replicate'].eq(replicate).to_numpy(dtype=bool)
            populations.append(spec.align(panel.X[mask],panel.var_names))
        a,b=populations
        first=evaluate_condition(spec,a,X_pred=b,seed=seed)
        second=evaluate_condition(spec,b,X_pred=a,seed=seed)
        records.append({'condition':condition,**{k:(first[k]+second[k])/2 for k in first}})
    return records

def save_diagnostics(run,method,split_name,seed,panel,pred,spec):
    records=[]; scores=[]; other=np.ones(spec.n_genes,dtype=bool); other[spec.program_idx]=False
    for condition in sorted(pred.obs['condition'].unique()):
        for rep in ['rep1','rep2']:
            real=panel[panel.obs['condition'].eq(condition)&panel.obs['replicate'].eq(rep)]
            generated=pred[pred.obs['condition'].eq(condition)&pred.obs['replicate'].eq(rep)]
            rec=evaluate_condition(spec,real.X,X_pred=generated.X,seed=0)
            records.append(dict(method=method,split=split_name,seed=seed,condition=condition,replicate=rep,**rec))
        for label,cells in [('real',panel[panel.obs['condition'].eq(condition)]),('generated',pred[pred.obs['condition'].eq(condition)])]:
            r=response_score(cells.X,spec)
            scores.append(pd.DataFrame({'condition':condition,'population':label,'replicate':cells.obs['replicate'].astype(str).values,'response_score':r}))
    out=run/'diagnostics'/method; out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(records).to_csv(out/f'{split_name}_s{seed}_well_metrics.csv',index=False)
    pd.concat(scores).to_csv(out/f'{split_name}_s{seed}_response_scores.csv.gz',index=False)
    distribution=[]
    for condition in sorted(pred.obs['condition'].unique()):
        real=dense(panel.X[panel.obs['condition'].eq(condition).to_numpy()])
        generated=dense(pred.X[pred.obs['condition'].eq(condition).to_numpy()])
        distribution.append(dict(condition=condition,rmse_mean_nonprogram=float(np.sqrt(np.mean((real[:,other].mean(0)-generated[:,other].mean(0))**2))),
          total_variance_ratio=float(generated.var(0).sum()/real.var(0).sum()),
          mean_gene_zero_fraction_real=float((real==0).mean()),mean_gene_zero_fraction_pred=float((generated==0).mean())))
    pd.DataFrame(distribution).to_csv(out/f'{split_name}_s{seed}_distribution_checks.csv',index=False)

def trial_options(config):
    trials={'cellflow':dict(method='base',weight=0.,h=0.1)}
    def token(v): return f'{v:g}'.replace('.','p')
    for weight in config['wrf_weight_grid']:
        for h in config['wrf_bandwidth_grid']:
            trials[f'wrf_l{token(weight)}_h{token(h)}']=dict(method='wrf',weight=weight,h=h)
    for weight in config['mmd_weight_grid']:
        trials[f'mmd_l{token(weight)}']=dict(method='mmd',weight=weight,h=0.1)
    return trials

def train_seed(run, seed, trial='cellflow'):
    import jax
    import optax
    from cellflow.model import CellFlow
    from cellflow.data._dataloader import TrainSampler
    from cellflow.utils import match_linear
    assert any(d.platform=='gpu' for d in jax.devices()),jax.devices()
    config=json.loads((run/'config.json').read_text())
    early_audit=trial=='cellflow_early'
    options=dict(method='base',weight=0.,h=0.1) if early_audit else trial_options(config)[trial]
    stop_after=3000 if early_audit else config['training_steps']
    validation_steps=[500,1000,2000,3000] if early_audit else config['validation_steps']
    directory=run/'trials'/trial/f'seed_{seed}'
    directory.mkdir(parents=True,exist_ok=False)
    save_json(directory/'trial_config.json',options)
    shutil.copyfile(__file__,directory/'training_source.py')
    shutil.copyfile(ROOT/'src/cellflow_population_loss.py',directory/'population_loss_source.py')
    save_json(directory/'source_hashes.json',dict(training=digest(__file__),population_loss=digest(ROOT/'src/cellflow_population_loss.py'),
              config=digest(run/'config.json'),stop_after=stop_after,validation_steps=validation_steps))
    panel=ad.read_h5ad(PANEL)
    train=model_data(panel[panel.obs['split'].isin(['train','control'])])
    val=panel[panel.obs['split'].isin(['val','control'])].copy()
    del panel
    assert train.n_obs==19398 and train.n_vars==2085
    train.obs[['condition','split','replicate']].to_csv(directory/'training_cell_ids.csv')
    cf=CellFlow(train,solver='otfm')
    cf.prepare_data(sample_rep='X',control_key='is_control',
                    perturbation_covariates={'drug':('drug',),'dose':('log_dose',)},
                    perturbation_covariate_reps={'drug':'drug_embeddings'},
                    split_covariates=['replicate'],max_combination_length=1)
    schedule=optax.warmup_cosine_decay_schedule(init_value=1e-5,peak_value=config['learning_rate_peak'],
                      warmup_steps=config['warmup_steps'],decay_steps=config['training_steps'],end_value=config['learning_rate_end'])
    optimizer=optax.chain(optax.clip_by_global_norm(5),optax.adamw(schedule,weight_decay=config['weight_decay']))
    cf.prepare_model(condition_mode='deterministic',pooling='mean',regularization=0.,
                     layers_before_pool={k:{'layer_type':'mlp','dims':config['condition_mlp_dims'],'dropout_rate':0.} for k in ['drug','dose']},
                     layers_after_pool={'layer_type':'mlp','dims':[128],'dropout_rate':0.},
                     condition_embedding_dim=config['condition_embedding_dim'],cond_output_dropout=config['condition_output_dropout'],
                     time_freqs=config['time_freqs'],time_encoder_dims=tuple(config['time_encoder_dims']),
                     hidden_dims=tuple(config['hidden_dims']),decoder_dims=tuple(config['decoder_dims']),
                     probability_path={'constant_noise':config['constant_noise']},
                     match_fn=functools.partial(match_linear,epsilon=config['ot_epsilon'],tau_a=1.,tau_b=1.),
                     optimizer=optimizer,solver_kwargs={'ema':config['ema']},seed=seed)
    params=sum(x.size for x in jax.tree.leaves(cf.solver.vf_state.params))
    save_json(directory/'model_info.json',dict(n_parameters=int(params),gpu=[str(d) for d in jax.devices()],
              seed=seed,job_id=os.environ.get('SLURM_JOB_ID'),hostname=os.uname().nodename,
              cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),slurm_step_gpus=os.environ.get('SLURM_STEP_GPUS')))
    sampler=TrainSampler(data=cf.train_data,batch_size=config['batch_size'])
    rng_np=np.random.default_rng(seed)
    rng_jax=jax.random.PRNGKey(seed)
    spec=EvalSpec.load(REPO/'artifacts/eval_spec_v1')
    if options['method']!='base':
        from cellflow_population_loss import make_population_step
        population_step=make_population_step(cf.solver,spec,options['method'],options['weight'],options['h'],config)
    best=(float('inf'),float('inf')); losses=[]; validations=[]; start=time.monotonic()
    learning_rates=np.asarray(schedule(np.arange(config['training_steps'])))
    log=open(directory/'training_loss.csv','w',buffering=1)
    log.write('step,total_loss,flow_matching_loss,auxiliary_loss,auxiliary_active,learning_rate,wall_seconds\n')
    for step in range(1,stop_after+1):
        rng_jax,step_rng=jax.random.split(rng_jax)
        batch=sampler.sample(rng_np)
        if options['method']=='base':
            loss=float(cf.solver.step_fn(step_rng,batch))
            fm,aux,active=loss,0.,False
        else:
            total,fm,aux,active=population_step(step_rng,batch)
            loss,fm,aux,active=float(total),float(fm),float(aux),bool(active)
        assert np.isfinite(loss),(step,loss)
        losses.append(loss)
        log.write(f'{step},{loss},{fm},{aux},{int(active)},{float(learning_rates[step-1])},{time.monotonic()-start}\n')
        if step==1 or step%1000==0:
            print('PROGRESS',trial,seed,step,'mean_loss',float(np.mean(losses[-1000:])),
                  'seconds',round(time.monotonic()-start,1),flush=True)
        if step in validation_steps:
            cf.solver.is_trained=True
            pred=predict(cf,train,config,'val',seed)
            records=evaluate_predictions(spec,val,pred,split='val',seed=0)
            summary=write_metrics(run,f'{trial}_step{step}','val',seed,records,checkpoint=step)
            score=tuple(summary[k]['mean'] for k in config['checkpoint_selection'])
            selected=score<best
            validations.append(dict(step=step,score=score,selected=selected,negative_fraction=float(pred.uns['negative_fraction_before_clipping'])))
            save_json(directory/'validation_history.json',validations)
            if selected:
                best=score
                cf.trainer.training_logs={'loss':losses.copy()}
                cf.save(str(directory),file_prefix='best',overwrite=True)
                pred.write_h5ad(directory/'preds_val.h5ad',compression='gzip')
                write_metrics(run,trial,'val',seed,records,checkpoint=step)
                save_json(directory/'selected_checkpoint.json',dict(trial=trial,seed=seed,step=step,selection_keys=config['checkpoint_selection'],selection_scores=score,
                        selected_on='validation only',test_evaluated=False))
            print('VALIDATION',trial,seed,step,'primary',score[0],'w1',score[1],'selected',selected,flush=True)
    log.close()
    save_json(directory/'training_complete.json',dict(seed=seed,steps=len(losses),elapsed_seconds=time.monotonic()-start,
                     first_1000_loss=float(np.mean(losses[:1000])),last_1000_loss=float(np.mean(losses[-1000:])),
                     all_losses_finite=True,versions={p:importlib.metadata.version(p) for p in ['cellflow-tools','jax','jaxlib','numpy','anndata','scanpy','scikit-learn']}))
    print('TRAINING_COMPLETE',trial,seed,flush=True)

def select_methods(run):
    config=json.loads((run/'config.json').read_text())
    options=trial_options(config)
    candidates=[]
    for trial,opt in options.items():
        directory=run/'trials'/trial/'seed_0'
        if (directory/'training_failed.json').exists():
            failure=json.loads((directory/'training_failed.json').read_text())
            candidates.append(dict(trial=trial,**opt,status='failed',failure=failure,step=failure['step'],selection_scores=None))
            continue
        assert (directory/'training_complete.json').exists(),trial
        selected=json.loads((directory/'selected_checkpoint.json').read_text())
        candidates.append(dict(trial=trial,**opt,status='complete',**{k:selected[k] for k in ['step','selection_scores']}))
    methods={'cellflow':'cellflow'}
    for method in ['wrf','mmd']:
        chosen=min([c for c in candidates if c['method']==method and c['status']=='complete'],key=lambda c:tuple(c['selection_scores']))
        methods[f'cellflow_{method}']=chosen['trial']
    save_json(run/'selected_methods.json',dict(methods=methods,candidates=candidates,selected_on='seed 0 validation only'))
    print('SELECTED_METHODS',json.dumps(methods),flush=True)

def merge_baseline_audit(run):
    """Select the baseline over both its original and earlier validation checkpoints."""
    changes=[]
    for seed in [0,1,2]:
        base=run/'trials/cellflow'/f'seed_{seed}'
        early=run/'trials/cellflow_early'/f'seed_{seed}'
        assert (base/'training_complete.json').exists() and (early/'training_complete.json').exists()
        original=json.loads((base/'selected_checkpoint.json').read_text())
        alternative=json.loads((early/'selected_checkpoint.json').read_text())
        assert not (run/'test_evaluation_started.json').exists()
        selected=tuple(alternative['selection_scores'])<tuple(original['selection_scores'])
        changes.append(dict(seed=seed,original=original,early=alternative,early_selected=selected))
        if selected:
            archive=base/'before_early_checkpoint_audit';archive.mkdir(exist_ok=False)
            for name in ['best_CellFlow.pkl','preds_val.h5ad','selected_checkpoint.json']:
                (base/name).rename(archive/name)
                shutil.copyfile(early/name,base/name)
            selected_meta={**alternative,'trial':'cellflow','source_trial':'cellflow_early'}
            save_json(base/'selected_checkpoint.json',selected_meta)
            file=run/'metrics/cellflow_early'/f'val_s{seed}'/'per_condition.jsonl'
            recs=[json.loads(line) for line in file.read_text().splitlines()]
            for rec in recs:
                for k in ['method','split','seed','spec_version','evaluation_commit','training_step']:
                    rec.pop(k,None)
            write_metrics(run,'cellflow','val',seed,recs,checkpoint=alternative['step'])
    save_json(run/'baseline_early_checkpoint_audit.json',dict(reason='Initial validation scores favored the earliest saved checkpoint; include earlier stopping points before any test evaluation.',
                  additional_steps=[500,1000,2000,3000],same_30000_step_learning_rate_schedule=True,seeds=changes))
    print('BASELINE_EARLY_AUDIT_MERGED',json.dumps(changes),flush=True)

def final_evaluation(run):
    from cellflow.model import CellFlow
    config=json.loads((run/'config.json').read_text())
    methods=json.loads((run/'selected_methods.json').read_text())['methods']
    assert all((run/'trials'/trial/f'seed_{s}'/'training_complete.json').exists() for trial in methods.values() for s in config['seeds'])
    # Freeze all model choices before loading test expression for scoring.
    choices={f'{method}/s{s}':json.loads((run/'trials'/trial/f'seed_{s}/selected_checkpoint.json').read_text())
             for method,trial in methods.items() for s in config['seeds']}
    frozen=dict(selected_models=choices,config_sha256=digest(run/'config.json'),
      model_sha256={f'{method}/s{s}':digest(run/'trials'/trial/f'seed_{s}'/'best_CellFlow.pkl') for method,trial in methods.items() for s in config['seeds']})
    if (run/'test_evaluation_started.json').exists():
        assert json.loads((run/'test_evaluation_started.json').read_text())==frozen,'Cannot change any selected model or settings after test evaluation starts.'
    else:
        save_json(run/'test_evaluation_started.json',frozen)
    panel=ad.read_h5ad(PANEL)
    train=model_data(panel[panel.obs['split'].isin(['train','control'])])
    spec=EvalSpec.load(REPO/'artifacts/eval_spec_v1')
    for method,trial in methods.items():
        for seed in config['seeds']:
            directory=run/'trials'/trial/f'seed_{seed}'
            if (directory/'preds_test.h5ad').exists():
                pred=ad.read_h5ad(directory/'preds_test.h5ad')
            else:
                cf=CellFlow.load(str(directory/'best_CellFlow.pkl'))
                pred=predict(cf,train,config,'test',seed)
                pred.write_h5ad(directory/'preds_test.tmp.h5ad',compression='gzip')
                (directory/'preds_test.tmp.h5ad').rename(directory/'preds_test.h5ad')
                del cf
            recs=evaluate_predictions(spec,panel,pred,split='test',seed=0)
            result=write_metrics(run,method,'test',seed,recs,checkpoint=choices[f'{method}/s{seed}']['step'])
            val_pred=ad.read_h5ad(directory/'preds_val.h5ad')
            val_recs=evaluate_predictions(spec,panel,val_pred,split='val',seed=0)
            write_metrics(run,method,'val',seed,val_recs,checkpoint=choices[f'{method}/s{seed}']['step'])
            save_diagnostics(run,method,'test',seed,panel,pred,spec)
            save_diagnostics(run,method,'val',seed,panel,val_pred,spec)
            # Seen-condition diagnostic for seed 0, never used to select a model.
            if seed==0:
                if (directory/'preds_train.h5ad').exists():
                    train_pred=ad.read_h5ad(directory/'preds_train.h5ad')
                else:
                    cf=CellFlow.load(str(directory/'best_CellFlow.pkl'))
                    train_pred=predict(cf,train,config,'train',seed)
                    train_pred.write_h5ad(directory/'preds_train.tmp.h5ad',compression='gzip')
                    (directory/'preds_train.tmp.h5ad').rename(directory/'preds_train.h5ad')
                    del cf
                write_metrics(run,method,'train',seed,evaluate_predictions(spec,panel,train_pred,split='train',seed=0),checkpoint=choices[f'{method}/s{seed}']['step'])
                save_diagnostics(run,method,'train',seed,panel,train_pred,spec)
                del train_pred
            print('TEST_RESULT',method,seed,json.dumps(result),flush=True)
            del pred,val_pred
    # Transparent reference methods on the same cells and the same frozen evaluator.
    for split_name in ['val','test']:
        control=prediction_controls(train,0,config['prediction_cells_per_condition'])
        _,metadata=condition_table(train,split_name)
        arrays=[]; frames=[]
        for condition,c in metadata.items():
            n=config['prediction_cells_per_condition']
            n1=int(round(n*c['n_rep1']/c['n_cells']))
            for rep,take in [('rep1',n1),('rep2',n-n1)]:
                src=control[control.obs['replicate'].eq(rep)][:take]
                arrays.append(dense(src.X))
                frames.append(pd.DataFrame({'condition':condition,'replicate':rep,'source_control_id':src.obs_names},index=[f'{condition}|{rep}|ctrl_{i}' for i in range(take)]))
        pred=ad.AnnData(np.vstack(arrays),obs=pd.concat(frames),var=train.var.copy())
        out=run/'reference_predictions';out.mkdir(exist_ok=True)
        pred.write_h5ad(out/f'unchanged_control_{split_name}.h5ad',compression='gzip')
        write_metrics(run,'unchanged_control',split_name,0,evaluate_predictions(spec,panel,pred,split=split_name,seed=0))
        write_metrics(run,'replicate_reference',split_name,0,replicate_reference_compatible(spec,panel,split_name,seed=0))
        rng=np.random.default_rng(2026)
        for method in ['nearest_dose','adjacent_dose_mixture']:
            arrays=[];frames=[]
            for condition,c in metadata.items():
                doses=np.sort(train.obs.loc[train.obs['drug'].eq(c['drug']),'dose'].astype(float).unique())
                target=c['dose_nM']; n=config['prediction_cells_per_condition']
                n1=int(round(n*c['n_rep1']/c['n_cells']))
                if method=='nearest_dose':
                    chosen=[float(doses[np.argmin(np.abs(np.log1p(doses)-np.log1p(target)))])]; weights=[1.]
                else:
                    low=float(doses[doses<target].max()); high=float(doses[doses>target].min())
                    upper=(np.log1p(target)-np.log1p(low))/(np.log1p(high)-np.log1p(low))
                    chosen=[low,high];weights=[1-upper,upper]
                for rep,take in [('rep1',n1),('rep2',n-n1)]:
                    dose_samples=rng.choice(chosen,take,p=weights)
                    ids=[]
                    for d in dose_samples:
                        pool=train.obs.index[train.obs['drug'].eq(c['drug'])&train.obs['dose'].eq(d)&train.obs['replicate'].eq(rep)].to_numpy()
                        ids.append(rng.choice(pool))
                    arrays.append(dense(train[ids].X))
                    frames.append(pd.DataFrame({'condition':condition,'replicate':rep,'source_cell_id':ids},index=[f'{condition}|{rep}|{method}_{i}' for i in range(take)]))
            prediction=ad.AnnData(np.vstack(arrays),obs=pd.concat(frames),var=train.var.copy())
            out=run/'reference_predictions';out.mkdir(exist_ok=True)
            prediction.write_h5ad(out/f'{method}_{split_name}.h5ad',compression='gzip')
            write_metrics(run,method,split_name,0,evaluate_predictions(spec,panel,prediction,split=split_name,seed=0))
    aggregate(run)
    save_json(run/'complete.json',dict(status='complete',job_id=os.environ.get('SLURM_JOB_ID'),seeds=config['seeds'],test_conditions=17,val_conditions=9))
    print('EXPERIMENT_COMPLETE',str(run),flush=True)

def aggregate(run):
    """Average seeds within each condition, then use the official condition bootstrap."""
    all_frames=[]
    for file in sorted((run/'metrics').glob('*/*/per_condition.jsonl')):
        all_frames.append(pd.read_json(file,lines=True))
    all_records=pd.concat(all_frames,ignore_index=True)
    all_records.to_csv(run/'all_per_condition_metrics.csv',index=False)
    final=all_records[all_records['method'].isin(['cellflow','cellflow_wrf','cellflow_mmd','unchanged_control','replicate_reference','nearest_dose','adjacent_dose_mixture'])]
    metric_keys=[c for c in final if c.startswith(('wrf_err','w1','mmd','rmse'))]
    overview=[]; summaries={}
    for (method,split_name),frame in final.groupby(['method','split'],observed=True):
        averaged=frame.groupby('condition',observed=True)[metric_keys].mean().reset_index()
        result=summarize(averaged.to_dict('records'))
        result.update(method=method,split=split_name,seeds=sorted(map(int,frame.seed.unique())),
                      interval='95% bootstrap over conditions after averaging training seeds within condition')
        summaries[f'{method}/{split_name}']=result
        for key in metric_keys:
            seed_means=frame.groupby('seed',observed=True)[key].mean()
            overview.append(dict(method=method,split=split_name,metric=key,mean=result[key]['mean'],
                                 ci95_low=result[key]['ci95'][0],ci95_high=result[key]['ci95'][1],
                                 training_seed_sd=float(seed_means.std(ddof=1)) if len(seed_means)>1 else 0.,
                                 n_seeds=len(seed_means),n_conditions=len(averaged)))
    save_json(run/'combined_summary.json',summaries)
    pd.DataFrame(overview).to_csv(run/'metric_summary.csv',index=False)

def initialize(run):
    run.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(ROOT/'src/cellflow_v1_config.json',run/'config.json')
    shutil.copyfile(__file__,run/'run_cellflow_v1.py')
    shutil.copyfile(ROOT/'src/cellflow_population_loss.py',run/'cellflow_population_loss.py')
    for name in ['check_population_loss.py','check_cellflow_v1_gpu.py','cellflow_worker.py','run_cellflow_workers.sh']:
        shutil.copyfile(ROOT/'src'/name,run/name)
    shutil.copytree(REPO/'artifacts/eval_spec_v1',run/'eval_spec_v1')
    shutil.copyfile(REPO/'configs/split_v1.json',run/'split_v1.json')
    save_json(run/'source_manifest.json',{str(p.relative_to(run)):digest(p) for p in run.rglob('*') if p.is_file()})
    audit_panel(run)
    config=json.loads((run/'config.json').read_text())
    options=trial_options(config)
    jobs=[dict(trial='cellflow',seed=s) for s in config['seeds']]
    jobs.extend(dict(trial=t,seed=0) for t in options if t!='cellflow')
    save_json(run/'initial_jobs.json',jobs)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('stage',choices=['initialize','train','select','merge-baseline-audit','evaluate','aggregate'])
    p.add_argument('--run',required=True,type=Path)
    p.add_argument('--seed',type=int)
    p.add_argument('--trial',default='cellflow')
    args=p.parse_args()
    if args.stage=='initialize': initialize(args.run)
    elif args.stage=='train': train_seed(args.run,args.seed,args.trial)
    elif args.stage=='select': select_methods(args.run)
    elif args.stage=='merge-baseline-audit': merge_baseline_audit(args.run)
    elif args.stage=='evaluate': final_evaluation(args.run)
    else: aggregate(args.run)
