"""Differentiable generated-population losses for the installed CellFlow solver.

Generation starts with controls and uses only the treatment condition. The true
treated cells enter the scalar target metric, never the generated trajectory.
"""
import jax
import jax.numpy as jnp
from ott.solvers import utils as solver_utils
from cellflow.solvers.utils import ema_update

def response_score_jax(x, program_idx, mu0, s0):
    p=jnp.maximum(x[:,program_idx],0.)
    # Algebraically log1p(sum(expm1(p))), without positive exponential arguments.
    # Continuous generators can briefly produce large positive values during
    # differentiable rollouts; direct expm1 overflows float32 and poisons gradients.
    shift=jax.lax.stop_gradient(jnp.max(p,axis=1,keepdims=True))
    scaled=jnp.exp(-shift[:,0])+jnp.sum(jnp.exp(p-shift)*(-jnp.expm1(-p)),axis=1)
    activity=shift[:,0]+jnp.log(scaled)
    return (mu0-activity)/(s0+1e-6)

def soft_fractions(x, program_idx, mu0, s0, taus, h):
    r=response_score_jax(x,program_idx,mu0,s0)
    return jax.nn.sigmoid((taus[None,:]-r[:,None])/h).mean(axis=0)

def mmd2_jax(a,b,mean,components,sigmas):
    za=(a-mean)@components.T
    zb=(b-mean)@components.T
    def kernel(x,y):
        d=jnp.maximum(jnp.sum(x*x,1)[:,None]+jnp.sum(y*y,1)[None,:]-2*x@y.T,0.)
        return jnp.mean(jnp.exp(-d[None,:,:]/(2*sigmas[:,None,None]**2)),axis=0)
    aa,bb,ab=kernel(za,za),kernel(zb,zb),kernel(za,zb)
    return ((aa.sum()-jnp.trace(aa))/(len(a)*(len(a)-1))
            +(bb.sum()-jnp.trace(bb))/(len(b)*(len(b)-1))-2*ab.mean())

def rollout(apply_fn,params,source,condition,condition_dim,n_steps):
    """Fixed-step Heun integration with gradients through every velocity evaluation."""
    dt=1./n_steps
    zeros=jnp.zeros((source.shape[0],condition_dim),dtype=source.dtype)
    def velocity(t,x):
        time=jnp.full((x.shape[0],1),t,dtype=x.dtype)
        return apply_fn({'params':params},time,x,condition,zeros,train=False)[0]
    def body(x,i):
        t=i*dt
        k1=velocity(t,x)
        k2=velocity(t+dt,x+dt*k1)
        return x+0.5*dt*(k1+k2),None
    generated,_=jax.lax.scan(jax.checkpoint(body),source,jnp.arange(n_steps))
    return generated

def make_population_step(solver,spec,method,weight,h,config):
    assert method in ['wrf','mmd']
    idx=jnp.asarray(spec.program_idx)
    taus=jnp.asarray(config.get('training_taus', spec.taus),dtype=jnp.float32)
    pmean=jnp.asarray(spec.pca_mean,dtype=jnp.float32)
    comps=jnp.asarray(spec.pca_components,dtype=jnp.float32)
    sigmas=jnp.asarray(spec.mmd_sigmas,dtype=jnp.float32)
    frequency=config['auxiliary_every']
    begin=config['auxiliary_start_step']
    n=config['auxiliary_batch_size']
    steps=config['auxiliary_ode_steps']

    @jax.jit
    def update(rng,state,time,source,target,condition,encoder_noise):
        active=(state.step>=begin)&(jnp.mod(state.step+1,frequency)==0)
        def loss(params):
            # Keep random-noise defaults and the base objective in their original precision.
            with jax.enable_x64(False):
                rng_flow,rng_encoder,rng_dropout=jax.random.split(rng,3)
                xt=solver.probability_path.compute_xt(rng_flow,time,source,target)
                vt,_,_=state.apply_fn({'params':params},time,xt,condition,encoder_noise=encoder_noise,
                                     rngs={'dropout':rng_dropout,'condition_encoder':rng_encoder})
                ut=solver.probability_path.compute_ut(time,xt,source,target)
                fm=jnp.mean((vt-ut)**2)
            def auxiliary(_):
                generated=rollout(state.apply_fn,params,source[:n].astype(jnp.float64),condition,
                                  solver.vf.condition_embedding_dim,steps)
                generated=jnp.maximum(generated,0.)
                if method=='wrf':
                    fp=soft_fractions(generated,idx,spec.mu0,spec.s0,taus,h)
                    ft=jax.lax.stop_gradient(soft_fractions(target.astype(jnp.float64),idx,spec.mu0,spec.s0,taus,h))
                    return jnp.mean((fp-ft)**2).astype(source.dtype)
                return mmd2_jax(generated,jax.lax.stop_gradient(target),pmean,comps,sigmas)
            aux=jax.lax.cond(active,auxiliary,lambda _:jnp.array(0.,dtype=source.dtype),operand=None)
            # Apply less frequently for cost, scaling to retain the stated expected loss weight.
            total=fm+weight*frequency*aux
            return total,(fm,aux,active)
        (total,metrics),grads=jax.value_and_grad(loss,has_aux=True)(state.params)
        return state.apply_gradients(grads=grads),(total,*metrics)

    def step(rng,batch):
        source,target=batch['src_cell_data'],batch['tgt_cell_data']
        condition=batch['condition']
        rng_resample,rng_time,rng_step,rng_encoder=jax.random.split(rng,4)
        n_batch=source.shape[0]
        time=solver.time_sampler(rng_time,n_batch)
        noise=jax.random.normal(rng_encoder,(n_batch,solver.vf.condition_embedding_dim))
        coupling=solver.match_fn(source,target)
        si,ti=solver_utils.sample_joint(rng_resample,coupling)
        with jax.enable_x64():
            solver.vf_state,metrics=update(rng_step,solver.vf_state,time,source[si],target[ti],condition,noise)
        if solver.ema==1.:
            solver.vf_state_inference=solver.vf_state
        else:
            solver.vf_state_inference=solver.vf_state_inference.replace(
                params=ema_update(solver.vf_state_inference.params,solver.vf_state.params,solver.ema))
        return metrics
    return step
