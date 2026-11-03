#!/bin/bash
set -euo pipefail
module load anaconda3/2024.10-1
source /ocean/projects/cis260266p/ywang179/response-fraction/project.env
source "$RESPONSE_FRACTION_ROOT/envs/cellflow/bin/activate"
export OMP_NUM_THREADS=5 OPENBLAS_NUM_THREADS=5 MKL_NUM_THREADS=5
export MPLCONFIGDIR="$RESPONSE_FRACTION_ROOT/cache/matplotlib"
export JAX_PLATFORMS=cpu
export CELLFLOW_RUN="$1"
python - <<'PY'
import hashlib, importlib, inspect, json, os, sys, types
from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd
root=Path(os.environ['RESPONSE_FRACTION_ROOT'])
run=Path(os.environ['CELLFLOW_RUN'])
sys.path.insert(0,str(root/'src'))
from run_cellflow_v1 import REPO,PANEL,replicate_reference_compatible
from sciplex_eval import EvalSpec
module=importlib.import_module('sciplex_eval.evaluate')
source=inspect.getsource(module._real)
old='m = (panel.obs["condition"] == cond).values'
assert old in source
patched=source.replace(old,'m = (panel.obs["condition"] == cond).to_numpy(dtype=bool, copy=True)')
scope=dict(vars(module));exec(patched,scope)
original_with_copy=types.FunctionType(module.replicate_reference.__code__,scope,
                                   argdefs=module.replicate_reference.__defaults__)
panel=ad.read_h5ad(PANEL);spec=EvalSpec.load(REPO/'artifacts/eval_spec_v1')
expected=original_with_copy(spec,panel,'val',seed=0)
actual=replicate_reference_compatible(spec,panel,'val',seed=0)
assert len(expected)==len(actual)==9
max_difference=0.
for a,b in zip(expected,actual):
    assert a['condition']==b['condition']
    for key in a:
        if key!='condition':
            np.testing.assert_allclose(a[key],b[key],rtol=0,atol=0)
            max_difference=max(max_difference,abs(a[key]-b[key]))
methods=json.loads((run/'selected_methods.json').read_text())['methods']
for trial in methods.values():
    for seed in [0,1,2]:
        directory=run/'trials'/trial/f'seed_{seed}'
        for split in (['val','test','train'] if seed==0 else ['val','test']):
            assert (directory/f'preds_{split}.h5ad').exists()
record=dict(pandas_version=pd.__version__,issue='Pinned replicate helper mutates a read-only NumPy boolean mask under pandas3.1.',
    fix='Copy only row masks in a local compatibility adapter; preserve sparse expression, bidirectional averaging, and all official metric functions.',
    unchanged_official_file_sha256=hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),
    check='Compared local adapter with original helper executed in an isolated namespace with only its mask copied. All returned fields match exactly on all9 validation conditions.',
    validation_conditions=9,max_absolute_difference=max_difference,passed=True,
    existing_predictions_only=True,no_model_or_hyperparameter_changes=True)
(run/'reference_compatibility_record.json').write_text(json.dumps(record,indent=2)+'\n')
(run/'checks/reference_compatibility_check.json').write_text(json.dumps(record,indent=2)+'\n')
print('REFERENCE_COMPATIBILITY_PASSED',max_difference,flush=True)
PY
python -u "$RESPONSE_FRACTION_ROOT/src/run_cellflow_v1.py" evaluate --run "$CELLFLOW_RUN"
python -u "$RESPONSE_FRACTION_ROOT/src/report_cellflow_v1.py" --run "$CELLFLOW_RUN"
SHARED_RESULTS="$(dirname "$RESPONSE_FRACTION_DATA")/results/$(basename "$CELLFLOW_RUN")"
python -u "$RESPONSE_FRACTION_ROOT/src/package_cellflow_results.py" --run "$CELLFLOW_RUN" --destination "$SHARED_RESULTS"
echo "FINALIZATION_COMPLETE $CELLFLOW_RUN"
