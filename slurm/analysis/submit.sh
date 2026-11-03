#!/bin/bash
# Submit the WRF analysis pipeline; this script only calls sbatch.
#
#   measured (CPU) -> extract (CPU array: chemCPA methods) -> analyze (CPU) -> figures (CPU)
#
# usage: slurm/analysis/submit.sh [VAR=value ...]
#   e.g. slurm/analysis/submit.sh                         # everything
#        slurm/analysis/submit.sh STAGES="analyze figures" # after copying CellFlow extracts in
#        slurm/analysis/submit.sh STAGES=figures           # redraw only
# Defaults are in defaults.sh. The resolved configuration is written to one env file
# per submission and passed to every job as a positional argument.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
for name in $(compgen -v SLURM_); do
  unset "$name"
done
source slurm/analysis/defaults.sh "$@"

logs="$ANALYSIS_ROOT/logs"
mkdir -p "$logs" "$ANALYSIS_ROOT/submissions"
env_file="$ANALYSIS_ROOT/submissions/$(date +%Y%m%d-%H%M%S).env"
for name in REPO VENV DATA_DIR PANEL ANALYSIS_ROOT STAGES N_BOOT; do
  printf '%s=%q\n' "$name" "${!name}"
done > "$env_file"

submit() {
  echo "+ sbatch --parsable $*" >&2
  local id
  id=$(sbatch --parsable "$@")
  echo "${id%%;*}"
}

configs=(configs/chemcpa_*.json)
after=()
ids=()
for stage in $STAGES; do
  case "$stage" in
    extract) extra=(--array="0-$(( ${#configs[@]} - 1 ))" --output="$logs/%x-%A_%a.out") ;;
    measured|analyze|figures) extra=(--output="$logs/%x-%j.out") ;;
    *) echo "unknown stage: $stage" >&2; exit 2 ;;
  esac
  id=$(submit "${after[@]}" "${extra[@]}" "slurm/analysis/$stage.sbatch" "$env_file")
  after=(--dependency="afterok:$id")
  ids+=("$stage=$id")
done
echo "${ids[*]}"
echo "env file: $env_file"
echo "logs: $logs"
