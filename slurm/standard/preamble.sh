# Shared preamble for the standard-benchmark jobs: configuration, venv, and where the job ran.
# Sourced with the job's arguments; the job body sets -euo pipefail.

source "$(dirname "${BASH_SOURCE[0]}")/defaults.sh" "$@"
export THREADS="${THREADS:-${SLURM_CPUS_PER_TASK:-4}}"
export OMP_NUM_THREADS="$THREADS" OPENBLAS_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS"
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg

cd "$REPO"
job_venv_var="${JOB_VENV_VAR:-VENV}"
active_venv="${!job_venv_var}"
source "$active_venv/bin/activate"
if [[ "$(command -v python)" != "$active_venv/bin/python" ]]; then
  echo "preamble: $active_venv/bin/python is not the active python" >&2
  return 1
fi

echo "== $(date -Is) host=$(hostname) job=${SLURM_JOB_ID:-none} array=${SLURM_ARRAY_JOB_ID:-}_${SLURM_ARRAY_TASK_ID:-} name=${SLURM_JOB_NAME:-}"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  echo "== request: $(scontrol show job -o "$SLURM_JOB_ID" | tr ' ' '\n' | grep -E '^(Partition|NodeList|NumCPUs|MinMemoryNode|TimeLimit|ReqTRES|TresPerNode)=' | tr '\n' ' ')"
fi
nvidia-smi -L 2>/dev/null || echo "== no GPU visible (CPU job)"
echo "== config: STD_FILE=$STD_FILE STD_ROOT=$STD_ROOT SEEDS='$SEEDS' FIT_GROUPS='$FIT_GROUPS' SPLITS='$SPLITS' STAGES='$STAGES' THREADS=$THREADS"
echo "== code: $(git rev-parse --short HEAD) on $(git rev-parse --abbrev-ref HEAD) + $(git status --porcelain | wc -l) uncommitted paths; python $(python -c 'import sys; print(sys.version.split()[0])')"

array_item() {
  local -a items
  read -ra items <<<"${!1}"
  local i="${SLURM_ARRAY_TASK_ID:?array_item needs an array job}"
  if (( i >= ${#items[@]} )); then
    echo "array index $i is outside $1='${!1}'" >&2
    return 1
  fi
  echo "${items[i]}"
}
