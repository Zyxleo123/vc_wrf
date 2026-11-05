# Shared preamble for the chemCPA job scripts: configuration, venv, and a record of
# where the job ran. Sourced with the job's arguments; the job body sets -euo pipefail.

source "$(dirname "${BASH_SOURCE[0]}")/defaults.sh" "$@"
export THREADS="${THREADS:-${SLURM_CPUS_PER_TASK:-4}}"
export OMP_NUM_THREADS="$THREADS" OPENBLAS_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS"
export SCIPLEX_PANEL="$PANEL" PYTHONUNBUFFERED=1

cd "$REPO"
source "$VENV/bin/activate"
if [[ "$(command -v python)" != "$VENV/bin/python" ]]; then
  echo "preamble: $VENV/bin/python is not the active python" >&2
  return 1
fi

echo "== $(date -Is) host=$(hostname) job=${SLURM_JOB_ID:-none} array=${SLURM_ARRAY_JOB_ID:-}_${SLURM_ARRAY_TASK_ID:-} name=${SLURM_JOB_NAME:-}"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  echo "== request: $(scontrol show job -o "$SLURM_JOB_ID" | tr ' ' '\n' | grep -E '^(Partition|NodeList|NumCPUs|TimeLimit|ReqTRES|TresPerNode)=' | tr '\n' ' ')"
fi
nvidia-smi -L 2>/dev/null || echo "== no GPU visible"
echo "== config: DATA_DIR=$DATA_DIR RUN_ROOT=$RUN_ROOT VARIANTS='$VARIANTS' SEEDS='$SEEDS' EVALS='$EVALS' REBUILD_DATA=$REBUILD_DATA THREADS=$THREADS"
echo "== code: $(git rev-parse --short HEAD) + $(git status --porcelain | wc -l) uncommitted paths; python $(python -c 'import sys; print(sys.version.split()[0])')"

# Set VARIANT and SEED from SLURM_ARRAY_TASK_ID, enumerating VARIANTS x SEEDS.
array_task() {
  local -a variants seeds
  read -ra variants <<<"$VARIANTS"
  read -ra seeds <<<"$SEEDS"
  local i="${SLURM_ARRAY_TASK_ID:?array_task needs an array job}"
  if (( i >= ${#variants[@]} * ${#seeds[@]} )); then
    echo "array index $i is outside VARIANTS x SEEDS" >&2
    return 1
  fi
  VARIANT="${variants[i / ${#seeds[@]}]}"
  SEED="${seeds[i % ${#seeds[@]}]}"
  echo "== task $i: VARIANT=$VARIANT SEED=$SEED"
}
