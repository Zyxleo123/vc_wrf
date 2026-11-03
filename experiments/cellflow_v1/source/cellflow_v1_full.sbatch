#!/bin/bash
# Reproduce the finalized CellFlow comparison against the existing frozen panel.
# Every submission writes to a new results directory; it does not overwrite a run.
#SBATCH --job-name=cellflow-v1
#SBATCH --account=cis260266p
#SBATCH --partition=GPU-shared
#SBATCH --gpus=v100-16:3
#SBATCH --nodes=1
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=5
#SBATCH --mem=60G
#SBATCH --time=04:00:00
#SBATCH --chdir=/ocean/projects/cis260266p/ywang179/response-fraction
#SBATCH --output=/ocean/projects/cis260266p/ywang179/response-fraction/logs/cellflow-full-%j.out
#SBATCH --error=/ocean/projects/cis260266p/ywang179/response-fraction/logs/cellflow-full-%j.err

set -euo pipefail
module load anaconda3/2024.10-1
source /ocean/projects/cis260266p/ywang179/response-fraction/project.env
source "$RESPONSE_FRACTION_ROOT/envs/cellflow/bin/activate"
export OMP_NUM_THREADS=5 OPENBLAS_NUM_THREADS=5 MKL_NUM_THREADS=5
export MPLCONFIGDIR="$RESPONSE_FRACTION_ROOT/cache/matplotlib"
export NUMBA_CACHE_DIR="$RESPONSE_FRACTION_ROOT/cache/numba"
export XDG_CACHE_HOME="$RESPONSE_FRACTION_ROOT/cache"
export XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda
unset SLURM_GPUS SLURM_GPUS_PER_NODE SLURM_GPUS_PER_TASK SLURM_GPUS_PER_SOCKET
RUN_DIR="$RESPONSE_FRACTION_ROOT/results/cellflow_vc_wrf_v1_${SLURM_JOB_ID}"
python -u "$RESPONSE_FRACTION_ROOT/src/run_cellflow_v1.py" initialize --run "$RUN_DIR"
python -m pip freeze > "$RUN_DIR/requirements.lock.txt"

wait_workers() {
  local failed=0
  for pid in "${worker_pids[@]}"; do wait "$pid" || failed=1; done
  if [ "$failed" -ne 0 ]; then echo 'A worker failed; inspect preserved logs.' >&2; exit 1; fi
}
for phase in initial; do
  worker_pids=()
  for worker in 0 1 2; do
    srun --exclusive --exact --nodes=1 --ntasks=1 --cpus-per-task=5 --gres=gpu:v100-16:1 --mem=20G \
      bash "$RESPONSE_FRACTION_ROOT/src/run_cellflow_workers.sh" "$RUN_DIR" "$phase" "$worker" \
      > "$RUN_DIR/${phase}_worker_${worker}.log" 2>&1 &
    worker_pids+=("$!")
  done
  wait_workers
done
worker_pids=()
for seed in 0 1 2; do
  srun --exclusive --exact --nodes=1 --ntasks=1 --cpus-per-task=5 --gres=gpu:v100-16:1 --mem=20G \
    python -u "$RESPONSE_FRACTION_ROOT/src/run_cellflow_v1.py" train --run "$RUN_DIR" --trial cellflow_early --seed "$seed" \
    > "$RUN_DIR/logs/cellflow_early_seed${seed}.log" 2>&1 &
  worker_pids+=("$!")
done
wait_workers
python -u "$RESPONSE_FRACTION_ROOT/src/run_cellflow_v1.py" merge-baseline-audit --run "$RUN_DIR"
python -u "$RESPONSE_FRACTION_ROOT/src/run_cellflow_v1.py" select --run "$RUN_DIR"
worker_pids=()
for worker in 0 1 2; do
  srun --exclusive --exact --nodes=1 --ntasks=1 --cpus-per-task=5 --gres=gpu:v100-16:1 --mem=20G \
    bash "$RESPONSE_FRACTION_ROOT/src/run_cellflow_workers.sh" "$RUN_DIR" repeat "$worker" \
    > "$RUN_DIR/repeat_worker_${worker}.log" 2>&1 &
  worker_pids+=("$!")
done
wait_workers
srun --exclusive --exact --nodes=1 --ntasks=1 --cpus-per-task=5 --gres=gpu:v100-16:1 --mem=20G \
  python -u "$RESPONSE_FRACTION_ROOT/src/check_validation_integration.py" --run "$RUN_DIR"
srun --exclusive --exact --nodes=1 --ntasks=1 --cpus-per-task=5 --gres=gpu:v100-16:1 --mem=20G \
  python -u "$RESPONSE_FRACTION_ROOT/src/run_cellflow_v1.py" evaluate --run "$RUN_DIR"
python -u "$RESPONSE_FRACTION_ROOT/src/report_cellflow_v1.py" --run "$RUN_DIR"
echo "COMPLETE $RUN_DIR"
