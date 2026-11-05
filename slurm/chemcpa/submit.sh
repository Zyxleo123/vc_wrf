#!/bin/bash
# Submit the chemCPA pipeline; this script only calls sbatch.
#
#   data (CPU) -> train (GPU array: variant x seed) -> evaluate (CPU array) -> report (CPU)
#
# usage: slurm/chemcpa/submit.sh [VAR=value ...]
#   e.g. slurm/chemcpa/submit.sh REBUILD_DATA=1
#        slurm/chemcpa/submit.sh SKIP_DATA=1 VARIANTS=nb_v1 SEEDS="0 1 2 3 4"
# Defaults are in defaults.sh. The resolved configuration is written to one env file
# per submission and passed to every job as a positional argument.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
for name in $(compgen -v SLURM_); do
  unset "$name"
done
source slurm/chemcpa/defaults.sh "$@"

read -ra variants <<<"$VARIANTS"
read -ra seeds <<<"$SEEDS"
for variant in "${variants[@]}"; do
  [[ -f "configs/chemcpa_$variant.json" ]] || { echo "missing configs/chemcpa_$variant.json" >&2; exit 2; }
done
last=$(( ${#variants[@]} * ${#seeds[@]} - 1 ))
logs="$RUN_ROOT/logs"
mkdir -p "$logs" "$RUN_ROOT/submissions"
env_file="$RUN_ROOT/submissions/$(date +%Y%m%d-%H%M%S).env"
for name in REPO VENV DATA_DIR PANEL CHEMICALS RUN_ROOT VARIANTS SEEDS EVALS COMPARE_NAME REBUILD_DATA; do
  printf '%s=%q\n' "$name" "${!name}"
done > "$env_file"

submit() {
  echo "+ sbatch --parsable $*" >&2
  local id
  id=$(sbatch --parsable "$@")
  echo "${id%%;*}"
}

after=()
data=skipped
if [[ "$SKIP_DATA" != 1 ]]; then
  data=$(submit --output="$logs/%x-%j.out" slurm/chemcpa/data.sbatch "$env_file")
  after=(--dependency="afterok:$data")
fi
train=$(submit "${after[@]}" --array="0-$last" --output="$logs/%x-%A_%a.out" slurm/chemcpa/train.sbatch "$env_file")
evaluate=$(submit --dependency="afterok:$train" --array="0-$last" --output="$logs/%x-%A_%a.out" \
  slurm/chemcpa/evaluate.sbatch "$env_file")
report=$(submit --dependency="afterok:$evaluate" --output="$logs/%x-%j.out" slurm/chemcpa/report.sbatch "$env_file")
echo "data=$data train=$train evaluate=$evaluate report=$report"
echo "env file: $env_file"
echo "logs: $logs"
