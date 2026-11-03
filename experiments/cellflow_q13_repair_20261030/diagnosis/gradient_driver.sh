#!/bin/bash
set -euo pipefail
module load anaconda3/2024.10-1
source /ocean/projects/cis260266p/ywang179/response-fraction/project.env
source "$RESPONSE_FRACTION_ROOT/envs/cellflow/bin/activate"
export RESPONSE_FRACTION_ROOT="$1"
export OMP_NUM_THREADS=5 OPENBLAS_NUM_THREADS=5 MKL_NUM_THREADS=5
export XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda,cpu
unset SLURM_GPUS SLURM_GPUS_PER_NODE SLURM_GPUS_PER_TASK SLURM_GPUS_PER_SOCKET
srun --jobid="$2" --exclusive --exact -N1 -n1 -c5 --gres=gpu:v100-16:1 --mem=20G \
 python -u "$RESPONSE_FRACTION_ROOT/src/check_gradient_repair.py" \
 --capture "$RESPONSE_FRACTION_ROOT/diagnosis_seed1/failure_state.pkl" \
 --out "$RESPONSE_FRACTION_ROOT/production/diagnosis/gradient_repair_check.json" \
 > "$RESPONSE_FRACTION_ROOT/logs/gradient_repair_check.log" 2>&1
