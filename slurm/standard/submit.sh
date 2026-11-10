#!/bin/bash
# Submit the standard sci-Plex3 benchmark; this script only calls sbatch.
#
#   prep (CPU)                         frozen split, evaluation spec, drug SMILES
#   embed (CPU, Uni-Mol env)  after prep     drug embeddings
#   teacher (CPU)             after prep, embed    Ours' condition response embedding
#   ours (GPU array: seed group)   after teacher   train + generate (CellFlow env)
#   doloris (GPU array: seed group) after embed    train + generate
#   score (CPU array: split)  after ours, doloris  DOLORIS metrics + MMD² for every method
#   report (CPU)              after score          tables in DOLORIS's format; run alone to redraw
#
# usage: slurm/standard/submit.sh [VAR=value ...]   e.g. STAGES="ours" SEEDS=0
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
for name in $(compgen -v SLURM_); do
  unset "$name"
done
source slurm/standard/defaults.sh "$@"
logs="$STD_ROOT/logs"
mkdir -p "$logs" "$STD_ROOT/submissions"
env_file=$(mktemp "$STD_ROOT/submissions/$(date +%Y%m%d-%H%M%S)-XXXX.env")
for name in REPO VENV CELLFLOW_VENV UNIMOL_VENV UNIMOL_WEIGHTS STD_FILE STD_ROOT SEEDS FIT_GROUPS SPLITS STAGES AFTER; do
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
    prep) ;;
    embed) [[ -n "${id[prep]:-}" ]] && deps=("${id[prep]}") ;;
    teacher) for s in prep embed; do [[ -n "${id[$s]:-}" ]] && deps+=("${id[$s]}"); done ;;
    ours) [[ -n "${id[teacher]:-}" ]] && deps=("${id[teacher]}") ;;
    doloris) for s in prep embed; do [[ -n "${id[$s]:-}" ]] && deps+=("${id[$s]}"); done ;;
    score) for s in ours doloris; do [[ -n "${id[$s]:-}" ]] && deps+=("${id[$s]}"); done ;;
    report) [[ -n "${id[score]:-}" ]] && deps=("${id[score]}") ;;
    *) echo "unknown stage: $stage" >&2; exit 2 ;;
  esac
  extra=(--output="$logs/%x-%j.out")
  case "$stage" in
    ours|doloris) extra=(--array="0-$(( ${#groups[@]} - 1 ))" --output="$logs/%x-%A_%a.out") ;;
    score) extra=(--array="0-$(( ${#splits[@]} - 1 ))" --output="$logs/%x-%A_%a.out") ;;
  esac
  [[ -n "$AFTER" ]] && deps+=("$AFTER")
  dep=()
  (( ${#deps[@]} )) && dep=(--dependency="afterok:$(IFS=:; echo "${deps[*]}")")
  id[$stage]=$(submit "${dep[@]}" "${extra[@]}" "slurm/standard/$stage.sbatch" "$env_file")
  echo "$stage=${id[$stage]}"
done
echo "env file: $env_file"
