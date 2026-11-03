"""Check raw gradients and actual optimizer updates on the captured failing batch."""
import argparse
import json
from pathlib import Path
import cloudpickle
import jax
import numpy as np
import optax
import run_cellflow_v1  # Register the pinned evaluator path before unpickling EvalSpec.
import cellflow_population_loss_fp64 as repaired

p=argparse.ArgumentParser();p.add_argument('--capture',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
with a.capture.open('rb') as stream:c=cloudpickle.load(stream)
solver=c['solver'];before=solver.vf_state;tx=before.tx;records=[]

def record(grads,device_norm,updates):
    arrays=[np.asarray(x,dtype=np.float64) for x in jax.tree.leaves(grads)]
    update_arrays=[np.asarray(x,dtype=np.float64) for x in jax.tree.leaves(updates)]
    records.append(dict(all_raw_gradients_finite=all(np.isfinite(x).all() for x in arrays),
        original_optimizer_gradient_norm=float(device_norm),
        independent_float64_gradient_norm=float(np.sqrt(sum(np.sum(x*x) for x in arrays))),
        max_absolute_raw_gradient=max(float(np.max(np.abs(x))) for x in arrays),
        all_updates_finite=all(np.isfinite(x).all() for x in update_arrays),
        optimizer_update_norm=float(np.sqrt(sum(np.sum(x*x) for x in update_arrays)))))

def update(grads,state,params=None,**kwargs):
    changes,new_state=tx.update(grads,state,params,**kwargs)
    jax.debug.callback(record,grads,optax.global_norm(grads),changes,ordered=True)
    return changes,new_state

solver.vf_state=before.replace(tx=optax.GradientTransformationExtraArgs(tx.init,update))
step=repaired.make_population_step(solver,c['spec'],'wrf',c['weight'],c['h'],c['config'])
values=step(c['rng'],c['batch']);jax.block_until_ready(values);jax.effects_barrier()
assert len(records)==1
result=records[0]
result.update(captured_seed=c['seed'],captured_step=c['step'],loss=float(values[0]),
    optimizer_unmodified=True,no_gradient_replacement=True)
result['passed']=bool(result['all_raw_gradients_finite'] and result['all_updates_finite'] and
    np.isfinite(result['original_optimizer_gradient_norm']) and result['optimizer_update_norm']>0 and np.isfinite(float(values[0])))
a.out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print('GRADIENT_REPAIR_CHECK',json.dumps(result),flush=True)
assert result['passed'],'Raw-gradient or optimizer overflow remains.'
