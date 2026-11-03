"""Replay unchanged training and capture the first nonfinite population loss."""
import argparse
import json
import os
from pathlib import Path
import shutil
import cloudpickle
import jax
import jax.numpy as jnp
import numpy as np
from ott.solvers import utils as solver_utils

import run_cellflow_v1 as base
import cellflow_population_loss as population

p=argparse.ArgumentParser()
p.add_argument('--seed',type=int,required=True)
p.add_argument('--run',type=Path,required=True)
p.add_argument('--config',type=Path,required=True)
a=p.parse_args()
a.run.mkdir(parents=True,exist_ok=False)
shutil.copyfile(a.config,a.run/'config.json')
factory=population.make_population_step

def stat(value):
    arr=np.asarray(value)
    finite=np.isfinite(arr)
    return dict(shape=list(arr.shape),finite=int(finite.sum()),total=int(arr.size),
                minimum=float(arr[finite].min()) if finite.any() else None,
                maximum=float(arr[finite].max()) if finite.any() else None)

def tree_stat(tree):
    leaves=jax.tree.leaves(tree)
    finite_max=[float(np.max(np.abs(np.asarray(x)[np.isfinite(np.asarray(x))]))) for x in leaves if np.isfinite(np.asarray(x)).any()]
    return dict(leaves=len(leaves),nonfinite_leaves=sum(not np.isfinite(np.asarray(x)).all() for x in leaves),
                max_absolute=max(finite_max) if finite_max else None)

class Captured(Exception): pass

def instrument(solver,spec,method,weight,h,config):
    original=factory(solver,spec,method,weight,h,config)
    def step(rng,batch):
        before=solver.vf_state
        inference=solver.vf_state_inference
        values=original(rng,batch)
        numbers=[float(x) for x in values]
        step_number=int(before.step)+1
        if not np.isfinite(numbers[:3]).all():
            after=solver.vf_state
            solver.vf_state=before
            solver.vf_state_inference=inference
            with (a.run/'failure_state.pkl').open('wb') as f:
                cloudpickle.dump(dict(solver=solver,spec=spec,rng=rng,batch=batch,config=config,
                                      weight=weight,h=h,seed=a.seed,step=step_number),f)
            result=dict(seed=a.seed,step=step_number,
                metrics=[x if np.isfinite(x) else str(x) for x in numbers],
                before_params=tree_stat(before.params),after_params=tree_stat(after.params),
                source=stat(batch['src_cell_data']),target=stat(batch['tgt_cell_data']))
            rres,rt,rstep,renc=jax.random.split(rng,4)
            coupling=solver.match_fn(batch['src_cell_data'],batch['tgt_cell_data'])
            si,ti=solver_utils.sample_joint(rres,coupling)
            source=batch['src_cell_data'][si][:config['auxiliary_batch_size']]
            target=batch['tgt_cell_data'][ti]
            condition=batch['condition']
            result['coupling']=stat(coupling)
            generated=jax.jit(lambda x: population.rollout(before.apply_fn,before.params,x,condition,
                solver.vf.condition_embedding_dim,config['auxiliary_ode_steps']))(source)
            result['generated']=stat(generated)
            clipped=jnp.maximum(generated,0.)
            idx=jnp.asarray(spec.program_idx)
            taus=jnp.asarray(config['training_taus'],dtype=jnp.float32)
            r=jax.jit(population.response_score_jax)(clipped,idx,spec.mu0,spec.s0)
            fp=population.soft_fractions(clipped,idx,spec.mu0,spec.s0,taus,h)
            ft=population.soft_fractions(target,idx,spec.mu0,spec.s0,taus,h)
            result.update(response=stat(r),predicted_fraction=stat(fp),target_fraction=stat(ft))
            # Compare the same expression to the stable float64 reference on the host.
            x=np.maximum(np.asarray(generated,dtype=np.float64)[:,spec.program_idx],0.)
            shift=x.max(1,keepdims=True)
            activity=shift[:,0]+np.log(np.exp(-shift[:,0])+(np.exp(x-shift)*(-np.expm1(-x))).sum(1))
            result['host_response']=stat((spec.mu0-activity)/(spec.s0+1e-6))
            result['forward_replay_aux']=float(jnp.mean((fp-ft)**2)) if np.isfinite(np.asarray(fp)).all() else 'nonfinite'
            np.savez_compressed(a.run/'failure_arrays.npz',generated=np.asarray(generated),
                source=np.asarray(source),target=np.asarray(target),response=np.asarray(r))
            # Trace the stages using the same Heun scheme to locate overflow.
            def trace(x):
                zeros=jnp.zeros((len(x),solver.vf.condition_embedding_dim),dtype=x.dtype)
                def velocity(t,y):
                    return before.apply_fn({'params':before.params},jnp.full((len(y),1),t,dtype=y.dtype),y,condition,zeros,train=False)[0]
                dt=1./config['auxiliary_ode_steps']
                def body(y,i):
                    k1=velocity(i*dt,y); predictor=y+dt*k1
                    k2=velocity((i+1)*dt,predictor); out=y+.5*dt*(k1+k2)
                    diagnostics=jnp.array([jnp.max(jnp.abs(y)),jnp.max(jnp.abs(k1)),jnp.max(jnp.abs(predictor)),jnp.max(jnp.abs(k2)),jnp.max(jnp.abs(out))])
                    return out,diagnostics
                return jax.lax.scan(body,x,jnp.arange(config['auxiliary_ode_steps']))[1]
            trace_values=np.asarray(jax.jit(trace)(source))
            result['rollout_absmax_by_stage']=[[float(v) if np.isfinite(v) else str(v) for v in row] for row in trace_values]
            base.save_json(a.run/'failure_diagnosis.json',result)
            print('FAILURE_CAPTURED',json.dumps(result),flush=True)
            raise Captured()
        if step_number>=4500:
            base.save_json(a.run/'no_failure_reproduced.json',dict(seed=a.seed,steps=step_number))
            raise Captured()
        return values
    return step

population.make_population_step=instrument
try:
    base.train_seed(a.run,a.seed,'wrf_l0p1_h0p3')
except Captured:
    pass
