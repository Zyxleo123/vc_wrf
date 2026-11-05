# Shared preamble for the analysis job scripts: configuration, venv, and a record of
# where the job ran. Sourced with the job's arguments; the job body sets -euo pipefail.

source "$(dirname "${BASH_SOURCE[0]}")/defaults.sh" "$@"
export THREADS="${THREADS:-${SLURM_CPUS_PER_TASK:-4}}"
export OMP_NUM_THREADS="$THREADS" OPENBLAS_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS"
export SCIPLEX_PANEL="$PANEL" PYTHONUNBUFFERED=1 MPLBACKEND=Agg

cd "$REPO"
source "$VENV/bin/activate"
if [[ "$(command -v python)" != "$VENV/bin/python" ]]; then
  echo "preamble: $VENV/bin/python is not the active python" >&2
  return 1
fi

echo "== $(date -Is) host=$(hostname) job=${SLURM_JOB_ID:-none} array=${SLURM_ARRAY_JOB_ID:-}_${SLURM_ARRAY_TASK_ID:-} name=${SLURM_JOB_NAME:-}"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  echo "== request: $(scontrol show job -o "$SLURM_JOB_ID" | tr ' ' '\n' | grep -E '^(Partition|NodeList|NumCPUs|MinMemoryNode|TimeLimit|ReqTRES|TresPerNode)=' | tr '\n' ' ')"
fi
nvidia-smi -L 2>/dev/null || echo "== no GPU visible (CPU job)"
echo "== config: PANEL=$PANEL ANALYSIS_ROOT=$ANALYSIS_ROOT STAGES='$STAGES' N_BOOT=$N_BOOT AUDIT_SAMPLING_SEEDS=$AUDIT_SAMPLING_SEEDS THREADS=$THREADS"
echo "== code: $(git rev-parse --short HEAD) on $(git rev-parse --abbrev-ref HEAD) + $(git status --porcelain | wc -l) uncommitted paths; python $(python -c 'import sys; print(sys.version.split()[0])')"
