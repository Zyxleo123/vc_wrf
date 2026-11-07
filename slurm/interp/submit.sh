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
#   teacher (CPU)                             mean-level dose-response teacher for CellFlow
#   cellflow_train (GPU array: variant x seed)    configs/interp/cellflow_<variant>.json; after teacher
#   cellflow_test (GPU array)  after cellflow_train  generation + frozen evaluator for each of $EVALS
#   cellflow_compare (CPU)     after cellflow_test   paired comparison with the base, published CellFlow, chemCPA
#   cellflow_fig_arms (CPU)                          every arm's ranking and training curves, from history.jsonl
#   cellflow_fig_collect (CPU) after cellflow_test   metric rows and cell profiles of $FIG_ARM on $FIG_SPLIT (needs dose_axes once)
#   cellflow_fig_draw (CPU)    after the two above   PNGs from those files; run alone to redraw
#   report_fig_draw (CPU)      after cellflow_fig_collect  report PNGs (one axis each, restricted baselines); run alone to redraw
#   cellflow_fig_umap (CPU)                          joint UMAP figure of $FIG_ARM on $FIG_SPLIT; reads the panel
#   dose_axes (CPU)                                  which PCs carry dose, from control and training cells only
#   dose_cellflow (GPU array: $DOSE_CELLFLOW x seed)  CellFlow populations over a dose grid, validation drugs only
#   dose_chemcpa (CPU array: $DOSE_CHEMCPA x seed)    the same for chemCPA
#   dose_collect (CPU)         after the three above  dose-response curves and gains
#   dose_draw (CPU)            after dose_collect    PNGs from those files; run alone to redraw
#   dose_gain_collect (CPU)    after dose_collect    the per-drug dose gain, its median, and the median and 75th-percentile drugs
#   dose_gain_draw (CPU)       after dose_gain_collect  PNGs from those files; run alone to redraw
#   teacher_viz_tests (CPU)                          the statistics below, on synthetic data with known answers
#   teacher_viz_collect (CPU)  after teacher_viz_tests  the teacher's vectors, per-coordinate meaning, neighbours and drug similarity
#   teacher_viz_draw (CPU)     after teacher_viz_collect  PNGs from those files; run alone to redraw
#   report (CPU)           after every other stage submitted with it
#
# usage: slurm/interp/submit.sh [VAR=value ...]
#   e.g. slurm/interp/submit.sh STAGES=diagnose
#        slurm/interp/submit.sh STAGES="stats populations report"
#        slurm/interp/submit.sh STAGES="chemcpa_train chemcpa_eval report" VARIANTS="nblog_v1 nbmean_l10"
#        slurm/interp/submit.sh STAGES=chemcpa_eval AFTER=12345   # wait (afterok) for job 12345 too
#        slurm/interp/submit.sh STAGES="cellflow_fig_collect cellflow_fig_draw" FIG_ARM=base FIG_SPLIT=val
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
for name in REPO VENV CELLFLOW_VENV CELLFLOW_CONFIG CELLFLOW_BASE CELLFLOW_LABELS DATA_DIR PANEL CHEMICALS INTERP_ROOT STAGES TASK_SETS CHEMCPA_ROOT BASE_CHEMCPA_ROOT VARIANTS SEEDS EVALS TEACHER_COMPONENTS STATS_SHARDS CHEMCPA_TASK_SETS SHIFT_PREDICTORS EVAL_TASKS AFTER POP_PREDICTORS POP_KINDS POP_SEEDS COMPARE_PRIMARY FIG_ARM FIG_SPLIT FIG_CHEMCPA DOSE_CELLFLOW DOSE_CHEMCPA; do
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
    diagnose|stats|chemcpa_train|teacher|tests|cellflow_tests|cellflow_fig_arms|cellflow_fig_umap|dose_axes|dose_cellflow|dose_chemcpa|teacher_viz_tests) ;;
    cellflow_train) [[ -n "${id[teacher]:-}" ]] && deps=("${id[teacher]}") ;;
    populations) [[ -n "${id[stats]:-}" ]] && deps=("${id[stats]}") ;;
    chemcpa_eval|chemcpa_test) [[ -n "${id[chemcpa_train]:-}" ]] && deps=("${id[chemcpa_train]}") ;;
    chemcpa_compare) [[ -n "${id[chemcpa_test]:-}" ]] && deps=("${id[chemcpa_test]}") ;;
    cellflow_test) [[ -n "${id[cellflow_train]:-}" ]] && deps=("${id[cellflow_train]}") ;;
    cellflow_compare|cellflow_fig_collect) [[ -n "${id[cellflow_test]:-}" ]] && deps=("${id[cellflow_test]}") ;;
    cellflow_fig_draw)
      for s in cellflow_fig_arms cellflow_fig_collect; do
        [[ -n "${id[$s]:-}" ]] && deps+=("${id[$s]}")
      done ;;
    report_fig_draw) [[ -n "${id[cellflow_fig_collect]:-}" ]] && deps=("${id[cellflow_fig_collect]}") ;;
    dose_collect)
      for s in dose_axes dose_cellflow dose_chemcpa; do
        [[ -n "${id[$s]:-}" ]] && deps+=("${id[$s]}")
      done ;;
    dose_draw) [[ -n "${id[dose_collect]:-}" ]] && deps=("${id[dose_collect]}") ;;
    dose_gain_collect) [[ -n "${id[dose_collect]:-}" ]] && deps=("${id[dose_collect]}") ;;
    dose_gain_draw) [[ -n "${id[dose_gain_collect]:-}" ]] && deps=("${id[dose_gain_collect]}") ;;
    teacher_viz_collect) [[ -n "${id[teacher_viz_tests]:-}" ]] && deps=("${id[teacher_viz_tests]}") ;;
    teacher_viz_draw) [[ -n "${id[teacher_viz_collect]:-}" ]] && deps=("${id[teacher_viz_collect]}") ;;
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
    cellflow_train|cellflow_test)
      (( n_runs > 0 )) || { echo "$stage needs VARIANTS and SEEDS" >&2; exit 2; }
      for v in "${variants[@]}"; do
        [[ -n "$CELLFLOW_CONFIG" || -f "configs/interp/cellflow_$v.json" ]] || { echo "missing configs/interp/cellflow_$v.json" >&2; exit 2; }
      done
      extra=(--array="0-$(( n_runs - 1 ))" --output="$logs/%x-%A_%a.out") ;;
    dose_cellflow|dose_chemcpa)
      list=DOSE_CELLFLOW
      [[ "$stage" == dose_chemcpa ]] && list=DOSE_CHEMCPA
      read -ra swept <<<"${!list}"
      extra=(--array="0-$(( ${#swept[@]} * ${#seeds[@]} - 1 ))" --output="$logs/%x-%A_%a.out") ;;
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
