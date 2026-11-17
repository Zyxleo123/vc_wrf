#!/bin/bash
# Submit the DOLORIS-sparsity-mask experiment for Ours; this script only calls sbatch.
#
#   mask (CPU)                         mask Ours' saved cells -> preds/{ours,ours_final}_masked
#   score (CPU array: split)  after mask    re-score every method (adds the masked rows) per split
#   report (CPU)              after score    rebuild the standard report (ours_masked beside ours)
#
# Reuses slurm/standard/{preamble.sh,defaults.sh,score.sbatch,report.sbatch} and the env-file
# convention. The DOLORIS runs (mask models) and the Ours preds must already be on disk.
#
# usage: slurm/standard/submit_mask.sh [VAR=value ...]   e.g. SEEDS="0 1 2" SPLITS="test ood val"
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
for name in $(compgen -v SLURM_); do
  unset "$name"
done
source slurm/standard/defaults.sh "$@"
logs="$STD_ROOT/logs"
mkdir -p "$logs" "$STD_ROOT/submissions"
env_file=$(mktemp "$STD_ROOT/submissions/$(date +%Y%m%d-%H%M%S)-mask-XXXX.env")
for name in REPO VENV CELLFLOW_VENV UNIMOL_VENV UNIMOL_WEIGHTS STD_FILE STD_ROOT SEEDS FIT_GROUPS SPLITS STAGES AFTER; do
  printf '%s=%q\n' "$name" "${!name}"
done > "$env_file"

submit() {
  echo "+ sbatch --parsable $*" >&2
  local id
  id=$(sbatch --parsable "$@") || { echo "sbatch failed: $*" >&2; return 1; }
  echo "${id%%;*}"
}

read -ra splits <<<"$SPLITS"
mask=$(submit --output="$logs/%x-%j.out" slurm/standard/mask_ours.sbatch "$env_file")
echo "mask=$mask"
score=$(submit --dependency="afterok:$mask" --array="0-$(( ${#splits[@]} - 1 ))" \
  --output="$logs/%x-%A_%a.out" slurm/standard/score.sbatch "$env_file")
echo "score=$score"
report=$(submit --dependency="afterok:$score" --output="$logs/%x-%j.out" slurm/standard/report.sbatch "$env_file")
echo "report=$report"
echo "env file: $env_file"
