#!/bin/bash
set -euo pipefail
module load anaconda3/2024.10-1
source /ocean/projects/cis260266p/ywang179/response-fraction/project.env
PARENT_ROOT="$RESPONSE_FRACTION_ROOT"
source "$PARENT_ROOT/envs/cellflow/bin/activate"
export RESPONSE_FRACTION_ROOT="$1"
export OMP_NUM_THREADS=5 OPENBLAS_NUM_THREADS=5 MKL_NUM_THREADS=5
export MPLCONFIGDIR="$RESPONSE_FRACTION_ROOT/cache/matplotlib" NUMBA_CACHE_DIR="$RESPONSE_FRACTION_ROOT/cache/numba" XDG_CACHE_HOME="$RESPONSE_FRACTION_ROOT/cache"
export XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda
unset SLURM_GPUS SLURM_GPUS_PER_NODE SLURM_GPUS_PER_TASK SLURM_GPUS_PER_SOCKET
mkdir -p "$RESPONSE_FRACTION_ROOT/cache" "$RESPONSE_FRACTION_ROOT/logs"
STEP=(srun --exclusive --exact -N1 -n1 -c5 --gres=gpu:v100-16:1 --mem=20G)
"${STEP[@]}" python -u "$RESPONSE_FRACTION_ROOT/src/run_repaired.py" initialize --root "$RESPONSE_FRACTION_ROOT" > "$RESPONSE_FRACTION_ROOT/logs/initialize.log" 2>&1
python -m pip freeze > "$RESPONSE_FRACTION_ROOT/requirements.lock.txt"
cmp "$RESPONSE_FRACTION_ROOT/requirements.lock.txt" "$PARENT_ROOT/results/cellflow_vc_wrf_v1_50003709/requirements.lock.txt"
for count in 3 13; do
  pids=()
  for seed in 0 1 2; do
    "${STEP[@]}" python -u "$RESPONSE_FRACTION_ROOT/src/run_repaired.py" train --root "$RESPONSE_FRACTION_ROOT" --count "$count" --seed "$seed" > "$RESPONSE_FRACTION_ROOT/logs/train_q${count}_seed${seed}.log" 2>&1 &
    pids+=("$!")
  done
  failed=0
  for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
  if [ "$failed" -ne 0 ]; then echo "A training run failed in q$count; test scoring not started." >&2; exit 1; fi
done
for count in 3 13; do
  "${STEP[@]}" python -u "$RESPONSE_FRACTION_ROOT/src/check_validation_integration.py" --run "$RESPONSE_FRACTION_ROOT/run$count" > "$RESPONSE_FRACTION_ROOT/logs/integration_q${count}.log" 2>&1
done
"${STEP[@]}" python -u "$RESPONSE_FRACTION_ROOT/src/run_repaired.py" evaluate --root "$RESPONSE_FRACTION_ROOT" > "$RESPONSE_FRACTION_ROOT/logs/evaluate.log" 2>&1
"${STEP[@]}" python -u "$RESPONSE_FRACTION_ROOT/src/report_repaired.py" --root "$RESPONSE_FRACTION_ROOT" > "$RESPONSE_FRACTION_ROOT/logs/report.log" 2>&1
"${STEP[@]}" python -u "$RESPONSE_FRACTION_ROOT/src/share_repaired.py" --root "$RESPONSE_FRACTION_ROOT" > "$RESPONSE_FRACTION_ROOT/logs/share.log" 2>&1
echo "REPAIRED_COMPARISON_COMPLETE job=$SLURM_JOB_ID root=$RESPONSE_FRACTION_ROOT"
