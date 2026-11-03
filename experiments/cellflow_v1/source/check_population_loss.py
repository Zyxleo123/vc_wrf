"""Cross-check the new differentiable metrics and actual model updates."""
import os
from pathlib import Path
import sys
import time
import jax
import jax.numpy as jnp
import numpy as np

root=Path(os.environ['RESPONSE_FRACTION_ROOT'])
sys.path.insert(0,str(root/'src/vc_wrf_1d8b173'))
from sciplex_eval import EvalSpec,response_score,mmd2_rbf
from cellflow_population_loss import response_score_jax,soft_fractions,mmd2_jax,make_population_step

spec=EvalSpec.load(root/'src/vc_wrf_1d8b173/artifacts/eval_spec_v1')
rng=np.random.default_rng(59)
x=rng.normal(.1,.4,(64,spec.n_genes)).astype(np.float32)
y=rng.normal(.15,.5,(80,spec.n_genes)).astype(np.float32)
np.testing.assert_allclose(np.asarray(response_score_jax(jnp.asarray(x),jnp.asarray(spec.program_idx),spec.mu0,spec.s0)),response_score(x,spec),atol=2e-6)
# Exact zeros and extreme transient generator values must have finite gradients.
for magnitude in [0.,1e-5,100.,1000.]:
 extreme=np.zeros((4,spec.n_genes),np.float32)
 extreme[:,spec.program_idx[0]]=magnitude
 fn=lambda v:response_score_jax(v,jnp.asarray(spec.program_idx),spec.mu0,spec.s0).sum()
 assert np.isfinite(np.asarray(jax.grad(fn)(jnp.asarray(extreme)))).all()
 assert np.isfinite(float(fn(jnp.asarray(extreme))))
 if magnitude<=100:
  np.testing.assert_allclose(np.asarray(response_score_jax(jnp.asarray(extreme),jnp.asarray(spec.program_idx),spec.mu0,spec.s0)),response_score(extreme,spec),atol=1e-5)
jmd=float(mmd2_jax(jnp.asarray(x),jnp.asarray(y),jnp.asarray(spec.pca_mean,dtype=jnp.float32),
 jnp.asarray(spec.pca_components,dtype=jnp.float32),jnp.asarray(spec.mmd_sigmas,dtype=jnp.float32)))
nmd=mmd2_rbf(spec.project(x),spec.project(y),spec.mmd_sigmas)
assert abs(jmd-nmd)<2e-6,(jmd,nmd)

# Response values near the primary threshold give an informative gradient check.
boundary=max(spec.mu0-spec.s0*spec.taus[1],0.05)
z=np.zeros((64,spec.n_genes),np.float32)
z[:,spec.program_idx[0]]=boundary
idx=jnp.asarray(spec.program_idx); taus=jnp.asarray(spec.taus,dtype=jnp.float32)
def f(v): return jnp.mean(soft_fractions(v,idx,spec.mu0,spec.s0,taus,.3))
grad=np.asarray(jax.grad(f)(jnp.asarray(z)))
assert np.isfinite(grad).all() and np.linalg.norm(grad)>0
assert abs(float(f(jnp.asarray(z+1e-3*grad)))-float(f(jnp.asarray(z-1e-3*grad))))>1e-8
print('METRIC_PARITY_AND_GRADIENT_PASSED',jmd,nmd,float(np.linalg.norm(grad)),flush=True)

import check_cellflow_v1_gpu as smoke
cf=smoke.cf
batch=smoke.sampler.sample(np.random.default_rng(72))
key=jax.random.PRNGKey(72)
# Flax TrainState and its JAX arrays are immutable; keep the pre-update values.
original=cf.solver.vf_state
inference=cf.solver.vf_state_inference
base_loss=float(cf.solver.step_fn(key,batch))
base_after=cf.solver.vf_state
cf.solver.vf_state=original; cf.solver.vf_state_inference=inference
cfg=smoke.c.copy(); cfg.update(auxiliary_start_step=100000,auxiliary_every=5)
joint=make_population_step(cf.solver,spec,'wrf',1.,.3,cfg)
joint_loss,fm,aux,active=joint(key,batch)
np.testing.assert_allclose(float(joint_loss),base_loss,atol=1e-6)
for a,b in zip(jax.tree.leaves(base_after.params),jax.tree.leaves(cf.solver.vf_state.params)):
 np.testing.assert_allclose(np.asarray(a),np.asarray(b),atol=2e-6,rtol=2e-6)
assert not bool(active) and float(aux)==0
print('INACTIVE_AUX_MATCHES_BASELINE',base_loss,flush=True)

for method in ['wrf','mmd']:
 cfg=smoke.c.copy(); cfg['auxiliary_start_step']=0
 joint=make_population_step(cf.solver,spec,method,1.,.3,cfg)
 start=time.monotonic(); active_losses=[]
 for i in range(100):
  key,k=jax.random.split(key)
  metrics=joint(k,smoke.sampler.sample(np.random.default_rng(i+1000)))
  assert all(np.isfinite(float(v)) for v in metrics[:3])
  if bool(metrics[3]): active_losses.append(float(metrics[2]))
 assert len(active_losses)==20
 print('POPULATION_LOSS_GPU_PASSED',method,'100_steps_seconds',time.monotonic()-start,
       'auxiliary_loss_mean',np.mean(active_losses),flush=True)
