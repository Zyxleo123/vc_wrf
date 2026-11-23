#!/bin/bash
# Submit the v2 arm of the standard benchmark; this script only calls sbatch.
#
#   teacher2 (CPU)                       condition response embedding v2 (analogue retrieval + magnitude)
#   ours_v2 (GPU array: seed group) after teacher2   train + generate with that embedding
#   calibrate (CPU)                 after ours_v2    marginal calibration sweep, val + ood
#   score (CPU array: split)        after ours_v2    rescore every method, the new one included
#   report (CPU)                    after score      rebuild the tables
#
# usage: slurm/standard/submit_v2.sh [VAR=value ...]   e.g. STAGES="teacher2 ours_v2" T2_REP=rdkit
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
for name in $(compgen -v SLURM_); do
  unset "$name"
done
export STAGES="${STAGES:-teacher2 ours_v2 score report}"
source slurm/standard/defaults.sh "$@"
export T2_REP="${T2_REP:-rdkit_morgan}" T2_FILE="${T2_FILE:-std_autodec_v2.npz}" OURS_TAG="${OURS_TAG:-ours_v2}"
export CAL_METHOD="${CAL_METHOD:-$OURS_TAG}" CAL_SPLITS="${CAL_SPLITS:-val ood}" CAL_WRITE="${CAL_WRITE:-}"
export GPU_FLAGS="${GPU_FLAGS:---partition=general --gres=gpu:a6000:1}"
logs="$STD_ROOT/logs"
mkdir -p "$logs" "$STD_ROOT/submissions"
env_file=$(mktemp "$STD_ROOT/submissions/$(date +%Y%m%d-%H%M%S)-v2-XXXX.env")
for name in REPO VENV CELLFLOW_VENV UNIMOL_VENV UNIMOL_WEIGHTS STD_FILE STD_ROOT SEEDS FIT_GROUPS SPLITS \
            STAGES AFTER T2_REP T2_FILE OURS_TAG CAL_METHOD CAL_SPLITS CAL_WRITE; do
  printf '%s=%q\n' "$name" "${!name}"
done > "$env_file"
submit() {
  echo "+ sbatch --parsable $*" >&2
  local id
  id=$(sbatch --parsable "$@") || { echo "sbatch failed: $*" >&2; return 1; }
  echo "${id%%;*}"
}
read -ra groups <<<"$FIT_GROUPS"
read -ra splits <<<"$SPLITS"
declare -A id
for stage in $STAGES; do
  deps=()
  case "$stage" in
    teacher2) ;;
    ours_v2) [[ -n "${id[teacher2]:-}" ]] && deps=("${id[teacher2]}") ;;
    calibrate) [[ -n "${id[ours_v2]:-}" ]] && deps=("${id[ours_v2]}") ;;
    score) for s2 in ours_v2 calibrate; do [[ -n "${id[$s2]:-}" ]] && deps+=("${id[$s2]}"); done ;;
    report) [[ -n "${id[score]:-}" ]] && deps=("${id[score]}") ;;
    *) echo "unknown stage: $stage" >&2; exit 2 ;;
  esac
  extra=(--output="$logs/%x-%j.out")
  case "$stage" in
    ours_v2) read -ra gpu <<<"$GPU_FLAGS"
             extra=("${gpu[@]}" --array="0-$(( ${#groups[@]} - 1 ))" --output="$logs/%x-%A_%a.out") ;;
    score) extra=(--array="0-$(( ${#splits[@]} - 1 ))" --output="$logs/%x-%A_%a.out") ;;
  esac
  [[ -n "$AFTER" ]] && deps+=("$AFTER")
  dep=()
  (( ${#deps[@]} )) && dep=(--dependency="afterok:$(IFS=:; echo "${deps[*]}")")
  id[$stage]=$(submit "${dep[@]}" "${extra[@]}" "slurm/standard/$stage.sbatch" "$env_file")
  echo "$stage=${id[$stage]}"
done
echo "env file: $env_file"
