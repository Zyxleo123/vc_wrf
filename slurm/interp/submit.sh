#!/bin/bash
# Submit dose-interpolation stages; this script only calls sbatch.
#
#   diagnose (CPU array)                      real-data and chemCPA diagnostics
#   stats (CPU array) -> stats_merge (CPU)    mean predictors ("simple networks") -> stats/
#   populations (CPU)      after stats        population generators scored on TASK_SETS
#   chemcpa_train (GPU array: variant x seed)
#   chemcpa_eval (CPU array) after chemcpa_train   population metrics, in-sample fit (sciplex_interp)
#   chemcpa_test (CPU array) after chemcpa_train   official evaluation, original pipeline ($EVALS)
#   chemcpa_compare (CPU)    after chemcpa_test    paired comparison with original chemCPA / CellFlow
#   report (CPU)           after every other stage submitted with it
#
# usage: slurm/interp/submit.sh [VAR=value ...]
#   e.g. slurm/interp/submit.sh STAGES=diagnose
#        slurm/interp/submit.sh STAGES="stats populations report"
#        slurm/interp/submit.sh STAGES="chemcpa_train chemcpa_eval report" VARIANTS="nblog_v1 nbmean_l10"
#        slurm/interp/submit.sh STAGES=chemcpa_eval AFTER=12345   # wait (afterok) for job 12345 too
# Defaults are in defaults.sh. The resolved configuration is written to one env file
# per submission and passed to every job as a positional argument.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
for name in $(compgen -v SLURM_); do
  unset "$name"
done
source slurm/interp/defaults.sh "$@"

logs="$INTERP_ROOT/logs"
mkdir -p "$logs" "$INTERP_ROOT/submissions"
env_file="$INTERP_ROOT/submissions/$(date +%Y%m%d-%H%M%S).env"
for name in REPO VENV DATA_DIR PANEL CHEMICALS INTERP_ROOT STAGES TASK_SETS CHEMCPA_ROOT BASE_CHEMCPA_ROOT VARIANTS SEEDS EVALS STATS_SHARDS CHEMCPA_TASK_SETS SHIFT_PREDICTORS EVAL_TASKS AFTER POP_PREDICTORS POP_KINDS POP_SEEDS COMPARE_PRIMARY; do
  printf '%s=%q\n' "$name" "${!name}"
done > "$env_file"

submit() {
  echo "+ sbatch --parsable $*" >&2
  local id
  id=$(sbatch --parsable "$@") || { echo "sbatch failed: $*" >&2; return 1; }
  echo "${id%%;*}"
}

read -ra variants <<<"$VARIANTS"
read -ra seeds <<<"$SEEDS"
n_runs=$(( ${#variants[@]} * ${#seeds[@]} ))
declare -A id
all=()
for stage in $STAGES; do
  deps=()
  case "$stage" in
    diagnose|stats|chemcpa_train|tests) ;;
    populations) [[ -n "${id[stats]:-}" ]] && deps=("${id[stats]}") ;;
    chemcpa_eval|chemcpa_test) [[ -n "${id[chemcpa_train]:-}" ]] && deps=("${id[chemcpa_train]}") ;;
    chemcpa_compare) [[ -n "${id[chemcpa_test]:-}" ]] && deps=("${id[chemcpa_test]}") ;;
    report) deps=("${all[@]}") ;;
    *) echo "unknown stage: $stage" >&2; exit 2 ;;
  esac
  extra=(--output="$logs/%x-%j.out")
  case "$stage" in
    diagnose|stats) extra=(--output="$logs/%x-%A_%a.out") ;;
    populations)
      read -ra ts <<<"$TASK_SETS"
      extra=(--array="0-$(( ${#ts[@]} - 1 ))" --output="$logs/%x-%A_%a.out") ;;
    chemcpa_train|chemcpa_eval|chemcpa_test)
      (( n_runs > 0 )) || { echo "$stage needs VARIANTS and SEEDS" >&2; exit 2; }
      for v in "${variants[@]}"; do
        [[ "$stage" != chemcpa_train || -f "configs/interp/chemcpa_$v.json" ]] || { echo "missing configs/interp/chemcpa_$v.json" >&2; exit 2; }
      done
      n_tasks=$n_runs
      [[ "$stage" != chemcpa_train ]] && n_tasks=$(( n_runs < EVAL_TASKS ? n_runs : EVAL_TASKS ))
      extra=(--array="0-$(( n_tasks - 1 ))" --output="$logs/%x-%A_%a.out") ;;
  esac
  [[ -n "$AFTER" ]] && deps+=("$AFTER")  # jobs of an earlier submission
  dep=()
  if (( ${#deps[@]} )); then
    dep=(--dependency="afterok:$(IFS=:; echo "${deps[*]}")")
  fi
  id[$stage]=$(submit "${dep[@]}" "${extra[@]}" "slurm/interp/$stage.sbatch" "$env_file")
  all+=("${id[$stage]}")
  echo "$stage=${id[$stage]}"
  if [[ "$stage" == stats ]]; then  # shards -> merge; downstream stages wait for the merge
    id[stats]=$(submit --dependency="afterok:${id[stats]}" --output="$logs/%x-%j.out" \
      slurm/interp/stats_merge.sbatch "$env_file")
    all+=("${id[stats]}")
    echo "stats_merge=${id[stats]}"
  fi
done
echo "env file: $env_file"
echo "logs: $logs"
