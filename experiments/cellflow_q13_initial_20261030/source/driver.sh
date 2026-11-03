#!/bin/bash
# Active command for a bounded Slurm allocation; no training runs on the login node.
set -euo pipefail
module load anaconda3/2024.10-1
source /ocean/projects/cis260266p/ywang179/response-fraction/project.env
CELLFLOW_PARENT_ROOT="$RESPONSE_FRACTION_ROOT"
source "$CELLFLOW_PARENT_ROOT/envs/cellflow/bin/activate"
export RESPONSE_FRACTION_ROOT="$1"
CELLFLOW_REFERENCE_RUN="$CELLFLOW_PARENT_ROOT/results/cellflow_vc_wrf_v1_50003709"
CELLFLOW_NEW_RUN="$RESPONSE_FRACTION_ROOT/run"
export OMP_NUM_THREADS=5 OPENBLAS_NUM_THREADS=5 MKL_NUM_THREADS=5
export MPLCONFIGDIR="$RESPONSE_FRACTION_ROOT/cache/matplotlib"
export NUMBA_CACHE_DIR="$RESPONSE_FRACTION_ROOT/cache/numba"
export XDG_CACHE_HOME="$RESPONSE_FRACTION_ROOT/cache"
export XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda
unset SLURM_GPUS SLURM_GPUS_PER_NODE SLURM_GPUS_PER_TASK SLURM_GPUS_PER_SOCKET
mkdir -p "$RESPONSE_FRACTION_ROOT/cache"
echo "ALLOCATION_START job=$SLURM_JOB_ID root=$RESPONSE_FRACTION_ROOT"
STEP=(srun --exclusive --exact --nodes=1 --ntasks=1 --cpus-per-task=5 --gres=gpu:v100-16:1 --mem=20G)
"${STEP[@]}" python -u "$RESPONSE_FRACTION_ROOT/src/run_threshold_comparison.py" initialize --run "$CELLFLOW_NEW_RUN" --baseline "$CELLFLOW_REFERENCE_RUN"
python -m pip freeze > "$CELLFLOW_NEW_RUN/requirements.lock.txt"
cmp "$CELLFLOW_REFERENCE_RUN/requirements.lock.txt" "$CELLFLOW_NEW_RUN/requirements.lock.txt"
"${STEP[@]}" python -u "$RESPONSE_FRACTION_ROOT/src/run_threshold_comparison.py" preflight --run "$CELLFLOW_NEW_RUN" --baseline "$CELLFLOW_REFERENCE_RUN" > "$CELLFLOW_NEW_RUN/logs/preflight.log" 2>&1
worker_pids=()
for seed in 0 1 2; do
  "${STEP[@]}" python -u "$RESPONSE_FRACTION_ROOT/src/run_threshold_comparison.py" train --run "$CELLFLOW_NEW_RUN" --baseline "$CELLFLOW_REFERENCE_RUN" --seed "$seed" > "$CELLFLOW_NEW_RUN/logs/train_seed${seed}.log" 2>&1 &
  worker_pids+=("$!")
done
failed=0
for pid in "${worker_pids[@]}"; do wait "$pid" || failed=1; done
if [ "$failed" -ne 0 ]; then echo 'Training failed; preserve artifacts and do not evaluate test.' >&2; exit 1; fi
"${STEP[@]}" python -u "$RESPONSE_FRACTION_ROOT/src/check_validation_integration.py" --run "$CELLFLOW_NEW_RUN" > "$CELLFLOW_NEW_RUN/logs/integration.log" 2>&1
"${STEP[@]}" python -u "$RESPONSE_FRACTION_ROOT/src/run_threshold_comparison.py" evaluate --run "$CELLFLOW_NEW_RUN" --baseline "$CELLFLOW_REFERENCE_RUN" > "$CELLFLOW_NEW_RUN/logs/evaluate.log" 2>&1
JAX_PLATFORMS=cpu "${STEP[@]}" python -u "$RESPONSE_FRACTION_ROOT/src/report_threshold_comparison.py" --run "$CELLFLOW_NEW_RUN" > "$CELLFLOW_NEW_RUN/logs/report.log" 2>&1
echo "COMPARISON_COMPLETE job=$SLURM_JOB_ID run=$CELLFLOW_NEW_RUN"
