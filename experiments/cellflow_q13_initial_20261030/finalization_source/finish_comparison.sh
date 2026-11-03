#!/bin/bash
# Finish the saved runs after the strict numerical replay check, without retraining.
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
STEP=(srun --exclusive --exact --nodes=1 --ntasks=1 --cpus-per-task=5 --gres=gpu:v100-16:1 --mem=20G)
mkdir -p "$CELLFLOW_NEW_RUN/finalization_source"
cp "$RESPONSE_FRACTION_ROOT/src/audit_controlled_settings.py" "$RESPONSE_FRACTION_ROOT/src/report_threshold_comparison.py" "$RESPONSE_FRACTION_ROOT/src/run_threshold_comparison.py" "$RESPONSE_FRACTION_ROOT/src/finish_comparison.sh" "$RESPONSE_FRACTION_ROOT/src/share_comparison.py" "$CELLFLOW_NEW_RUN/finalization_source/"
if [ ! -f "$CELLFLOW_NEW_RUN/checks/controlled_settings_and_numerics.json" ]; then
  "${STEP[@]}" python -u "$RESPONSE_FRACTION_ROOT/src/audit_controlled_settings.py" --run "$CELLFLOW_NEW_RUN" --baseline "$CELLFLOW_REFERENCE_RUN" > "$CELLFLOW_NEW_RUN/logs/controlled_settings_audit.log" 2>&1
fi
if ! python -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1]))["completed_seeds"] else 1)' "$CELLFLOW_NEW_RUN/training_outcomes.json"; then
  "${STEP[@]}" python -u "$RESPONSE_FRACTION_ROOT/src/share_comparison.py" --run "$CELLFLOW_NEW_RUN" --destination /ocean/projects/cis260266p/shared/response-fraction/results/cellflow_q13_20261030
  echo "ALL_SEEDS_FAILED_UNCHANGED_SETTINGS job=$SLURM_JOB_ID run=$CELLFLOW_NEW_RUN"
  exit 0
fi
if [ ! -f "$CELLFLOW_NEW_RUN/test_evaluation_started.json" ]; then
  "${STEP[@]}" python -u "$RESPONSE_FRACTION_ROOT/src/check_validation_integration.py" --run "$CELLFLOW_NEW_RUN" > "$CELLFLOW_NEW_RUN/logs/integration.log" 2>&1
fi
if [ ! -f "$CELLFLOW_NEW_RUN/evaluation_complete.json" ]; then
  "${STEP[@]}" python -u "$RESPONSE_FRACTION_ROOT/src/run_threshold_comparison.py" evaluate --run "$CELLFLOW_NEW_RUN" --baseline "$CELLFLOW_REFERENCE_RUN" > "$CELLFLOW_NEW_RUN/logs/evaluate.log" 2>&1
fi
JAX_PLATFORMS=cpu "${STEP[@]}" python -u "$RESPONSE_FRACTION_ROOT/src/report_threshold_comparison.py" --run "$CELLFLOW_NEW_RUN" > "$CELLFLOW_NEW_RUN/logs/report.log" 2>&1
"${STEP[@]}" python -u "$RESPONSE_FRACTION_ROOT/src/share_comparison.py" --run "$CELLFLOW_NEW_RUN" --destination /ocean/projects/cis260266p/shared/response-fraction/results/cellflow_q13_20261030
echo "COMPARISON_COMPLETE job=$SLURM_JOB_ID run=$CELLFLOW_NEW_RUN"
