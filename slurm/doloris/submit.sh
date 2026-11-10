#!/bin/bash
# Submit the DOLORIS reproduction; this script only calls sbatch.
#
#   tests (CPU)                         unit tests of the port and the shared metrics
#   parity (GPU)                        the port against the released code (tests/parity_doloris.py)
#   embed (CPU, Uni-Mol env)            drug embeddings
#   fit (GPU array: seed group) after embed     per seed: mask + diffusion model (checkpoints every
#                                               save_every), validation RMSE per checkpoint, cells for both rules
#   score (CPU array: split)   after fit        frozen evaluator + DOLORIS metrics for every method
#   report (CPU)               after score      tables; run alone to redraw
#
# usage: slurm/doloris/submit.sh [VAR=value ...]
#   e.g. slurm/doloris/submit.sh                                    # everything, configs/doloris_v1.json
#        slurm/doloris/submit.sh CONFIG_NAME=smoke SEEDS=0 FIT_GROUPS=0 SPLITS=val STAGES="fit score report"
#        slurm/doloris/submit.sh STAGES=report
# Defaults are in defaults.sh. The resolved configuration is written to one env file per
# submission and passed to every job as a positional argument.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
for name in $(compgen -v SLURM_); do
  unset "$name"
done
source slurm/doloris/defaults.sh "$@"
[[ -f "configs/doloris_$CONFIG_NAME.json" ]] || { echo "missing configs/doloris_$CONFIG_NAME.json" >&2; exit 2; }

logs="$DOLORIS_ROOT/logs"
mkdir -p "$logs" "$DOLORIS_ROOT/submissions"
# unique per submission: two submissions in the same second must not share (and overwrite) one file
env_file=$(mktemp "$DOLORIS_ROOT/submissions/$(date +%Y%m%d-%H%M%S)-XXXX.env")
for name in REPO VENV UNIMOL_VENV UNIMOL_WEIGHTS DOLORIS_SRC PYG_SITE DATA_DIR PANEL DOLORIS_ROOT CONFIG_NAME SEEDS FIT_GROUPS SPLITS SMILES_RUN STAGES AFTER; do
  printf '%s=%q\n' "$name" "${!name}"
done > "$env_file"

submit() {
  echo "+ sbatch --parsable $*" >&2
  local id
  id=$(sbatch --parsable "$@") || { echo "sbatch failed: $*" >&2; return 1; }
  echo "${id%%;*}"
}

read -ra groups <<<"$FIT_GROUPS"
[[ "$(tr ',' ' ' <<<"$FIT_GROUPS" | xargs -n1 | sort | xargs)" == "$(xargs -n1 <<<"$SEEDS" | sort | xargs)" ]] || {
  echo "FIT_GROUPS='$FIT_GROUPS' must cover SEEDS='$SEEDS' exactly" >&2; exit 2; }
read -ra splits <<<"$SPLITS"
declare -A id
for stage in $STAGES; do
  deps=()
  case "$stage" in
    tests|parity|embed) ;;
    fit) [[ -n "${id[embed]:-}" ]] && deps=("${id[embed]}") ;;
    score) [[ -n "${id[fit]:-}" ]] && deps=("${id[fit]}") ;;
    report) [[ -n "${id[score]:-}" ]] && deps=("${id[score]}") ;;
    *) echo "unknown stage: $stage" >&2; exit 2 ;;
  esac
  extra=(--output="$logs/%x-%j.out")
  case "$stage" in
    fit) extra=(--array="0-$(( ${#groups[@]} - 1 ))" --output="$logs/%x-%A_%a.out") ;;
    score) extra=(--array="0-$(( ${#splits[@]} - 1 ))" --output="$logs/%x-%A_%a.out") ;;
  esac
  [[ -n "$AFTER" ]] && deps+=("$AFTER")
  dep=()
  if (( ${#deps[@]} )); then
    dep=(--dependency="afterok:$(IFS=:; echo "${deps[*]}")")
  fi
  id[$stage]=$(submit "${dep[@]}" "${extra[@]}" "slurm/doloris/$stage.sbatch" "$env_file")
  echo "$stage=${id[$stage]}"
done
echo "env file: $env_file"
echo "logs: $logs"
