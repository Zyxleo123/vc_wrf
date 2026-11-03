"""Small synthetic integration check of the full-gene CellFlow configuration."""
import functools
import json
import os
from pathlib import Path
import time
import anndata as ad
import diffrax
import jax
import numpy as np
import optax
import pandas as pd
from cellflow.model import CellFlow
from cellflow.utils import match_linear
from cellflow.data._dataloader import TrainSampler

root=Path(os.environ['RESPONSE_FRACTION_ROOT'])
c=json.loads((root/'src/cellflow_v1_config.json').read_text())
rng=np.random.default_rng(417)
X=np.log1p(rng.poisson(0.25,(1024,2085))).astype(np.float32)
obs=pd.DataFrame({'is_control':[True]*512+[False]*512,
 'drug':['Vehicle']*512+['synthetic']*512,'log_dose':[0.]*512+[1.]*512,
 'replicate':['rep1']*1024},index=[str(i) for i in range(1024)])
a=ad.AnnData(X,obs=obs)
a.uns['drug_embeddings']={'Vehicle':np.zeros(26,dtype=np.float32),'synthetic':np.eye(26,dtype=np.float32)[0]}
cf=CellFlow(a,solver='otfm')
cf.prepare_data(sample_rep='X',control_key='is_control',
 perturbation_covariates={'drug':('drug',),'dose':('log_dose',)},
 perturbation_covariate_reps={'drug':'drug_embeddings'},split_covariates=['replicate'],max_combination_length=1)
schedule=optax.warmup_cosine_decay_schedule(1e-5,c['learning_rate_peak'],c['warmup_steps'],c['training_steps'],c['learning_rate_end'])
cf.prepare_model(condition_mode='deterministic',pooling='mean',regularization=0.,
 layers_before_pool={k:{'layer_type':'mlp','dims':c['condition_mlp_dims'],'dropout_rate':0.} for k in ['drug','dose']},
 layers_after_pool={'layer_type':'mlp','dims':[128],'dropout_rate':0.},
 condition_embedding_dim=c['condition_embedding_dim'],cond_output_dropout=c['condition_output_dropout'],
 time_freqs=c['time_freqs'],time_encoder_dims=tuple(c['time_encoder_dims']),hidden_dims=tuple(c['hidden_dims']),
 decoder_dims=tuple(c['decoder_dims']),probability_path={'constant_noise':c['constant_noise']},
 match_fn=functools.partial(match_linear,epsilon=c['ot_epsilon'],tau_a=1.,tau_b=1.),
 optimizer=optax.chain(optax.clip_by_global_norm(5),optax.adamw(schedule,weight_decay=c['weight_decay'])),
 solver_kwargs={'ema':c['ema']},seed=0)
sampler=TrainSampler(cf.train_data,batch_size=c['batch_size'])
key=jax.random.PRNGKey(417)
start=time.monotonic()
for i in range(1000):
 key,k=jax.random.split(key)
 loss=float(cf.solver.step_fn(k,sampler.sample(rng)))
 assert np.isfinite(loss)
 if i in [0,99,999]: print('SYNTHETIC_STEP',i+1,loss,'elapsed',time.monotonic()-start,flush=True)
cf.solver.is_trained=True
cov=a.obs.iloc[[512]].copy(); cov['target']='synthetic'
out=cf.predict(a[:512],cov,condition_id_key='target',solver=diffrax.Heun(),
 stepsize_controller=diffrax.ConstantStepSize(),dt0=1/64,max_steps=128)
assert np.asarray(out['synthetic']).shape==(512,2085)
assert np.isfinite(out['synthetic']).all()
print('SYNTHETIC_CHECK_PASSED',len(jax.tree.leaves(cf.solver.vf_state.params)),flush=True)
