#!/bin/bash
set -euo pipefail
module load anaconda3/2024.10-1
source /ocean/projects/cis260266p/ywang179/response-fraction/project.env
source "$RESPONSE_FRACTION_ROOT/envs/cellflow/bin/activate"
export OMP_NUM_THREADS=5 OPENBLAS_NUM_THREADS=5 MKL_NUM_THREADS=5
export MPLCONFIGDIR="$RESPONSE_FRACTION_ROOT/cache/matplotlib"
export NUMBA_CACHE_DIR="$RESPONSE_FRACTION_ROOT/cache/numba"
export XDG_CACHE_HOME="$RESPONSE_FRACTION_ROOT/cache"
export XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda
RUN_DIR="$1"
PHASE="$2"
WORKER="$3"
python -u "$RESPONSE_FRACTION_ROOT/src/cellflow_worker.py" --run "$RUN_DIR" --phase "$PHASE" --worker "$WORKER" --workers 3
