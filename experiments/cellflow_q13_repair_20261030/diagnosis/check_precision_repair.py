"""Validate the numerical repair on a captured failing batch before any rerun."""
import argparse
import json
from pathlib import Path
import cloudpickle
import jax
import jax.numpy as jnp
import numpy as np
from ott.solvers import utils as solver_utils

import cellflow_population_loss as old
import cellflow_population_loss_fp64 as repaired
from run_cellflow_v1 import save_json, digest

p=argparse.ArgumentParser()
p.add_argument('--capture',required=True,type=Path)
p.add_argument('--out',required=True,type=Path)
a=p.parse_args()
with a.capture.open('rb') as f: context=cloudpickle.load(f)
solver=context['solver']; spec=context['spec']; config=context['config']
before=solver.vf_state; inference=solver.vf_state_inference
rng=context['rng']; batch=context['batch']

def reset():
    solver.vf_state=before;solver.vf_state_inference=inference

def finite(tree):
    return all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(tree))

original=old.make_population_step(solver,spec,'wrf',context['weight'],context['h'],config)
got=original(rng,batch)
assert not np.isfinite([float(x) for x in got[:3]]).all(), 'Captured failure must reproduce.'
original_values=[float(x) if np.isfinite(float(x)) else str(float(x)) for x in got]
reset()
fixed=repaired.make_population_step(solver,spec,'wrf',context['weight'],context['h'],config)
values=fixed(rng,batch)
assert all(np.isfinite(float(x)) for x in values)
assert finite(solver.vf_state.params) and finite(solver.vf_state.opt_state)
assert all(np.asarray(x).dtype==np.float32 for x in jax.tree.leaves(solver.vf_state.params))
fixed_values=[float(x) for x in values]

# The inactive branch must preserve the original flow-matching update.
inactive={**config,'auxiliary_start_step':100000}
reset();off_old=old.make_population_step(solver,spec,'wrf',context['weight'],context['h'],inactive)
old_values=off_old(rng,batch);old_state=solver.vf_state
reset();off_fixed=repaired.make_population_step(solver,spec,'wrf',context['weight'],context['h'],inactive)
fixed_off_values=off_fixed(rng,batch)
np.testing.assert_allclose(np.asarray(old_values),np.asarray(fixed_off_values),rtol=1e-6,atol=1e-7)
for x,y in zip(jax.tree.leaves(old_state.params),jax.tree.leaves(solver.vf_state.params)):
    np.testing.assert_allclose(np.asarray(x),np.asarray(y),rtol=2e-6,atol=2e-6)
reset()

# Direct float64 rollout of the failed batch must remain finite without clipping or skipped cells.
rres,_,_,_=jax.random.split(rng,4)
coupling=solver.match_fn(batch['src_cell_data'],batch['tgt_cell_data'])
si,ti=solver_utils.sample_joint(rres,coupling)
source=batch['src_cell_data'][si][:config['auxiliary_batch_size']]
with jax.enable_x64():
    generated=jax.jit(lambda x: repaired.rollout(before.apply_fn,before.params,x,batch['condition'],
        solver.vf.condition_embedding_dim,config['auxiliary_ode_steps']))(source.astype(jnp.float64))
assert np.isfinite(np.asarray(generated)).all()

# Score and gradient parity on ordinary values, plus finite gradients at extreme values.
idx=jnp.asarray(spec.program_idx);taus=jnp.asarray(config['training_taus'])
generator=np.random.default_rng(710)
normal=generator.uniform(0,3,(16,spec.n_genes)).astype(np.float32)
def score(x): return repaired.soft_fractions(x,idx,spec.mu0,spec.s0,taus,context['h']).sum()
f32=float(score(jnp.asarray(normal)))
grad32=np.asarray(jax.grad(score)(jnp.asarray(normal)))
with jax.enable_x64():
    f64=float(score(jnp.asarray(normal,dtype=jnp.float64)))
    grad64=np.asarray(jax.grad(score)(jnp.asarray(normal,dtype=jnp.float64)))
    np.testing.assert_allclose(f32,f64,atol=2e-6,rtol=2e-6)
    np.testing.assert_allclose(grad32,grad64,atol=2e-6,rtol=2e-6)
    for magnitude in [0.,1e-5,100.,1e40,1e100]:
        x=np.zeros((4,spec.n_genes),dtype=np.float64);x[:,spec.program_idx[0]]=magnitude
        assert np.isfinite(float(score(jnp.asarray(x))))
        assert np.isfinite(np.asarray(jax.grad(score)(jnp.asarray(x)))).all()
result=dict(passed=True,captured_seed=context['seed'],captured_step=context['step'],
    original_metrics=original_values,repaired_metrics=fixed_values,
    repaired_rollout_max_abs=float(np.max(np.abs(np.asarray(generated)))),
    model_parameter_dtype='float32',optimizer_state_finite=True,
    inactive_flow_matching_update_matches=True,score_and_gradient_parity_passed=True,
    extreme_score_gradients_finite=True,auxiliary_ode_steps=config['auxiliary_ode_steps'],
    weight=context['weight'],h=context['h'],capture_sha256=digest(a.capture),
    repair='Use float64 for differentiable auxiliary trajectory and response score; cast final bounded scalar to original loss dtype. No clipping, skipped batch, or altered hyperparameter.')
save_json(a.out,result)
print('REPAIR_VERIFIED',json.dumps(result),flush=True)
