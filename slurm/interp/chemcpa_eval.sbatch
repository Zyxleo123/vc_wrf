#!/bin/bash
#SBATCH --job-name=interp-chemcpa-eval
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
# Score chemCPA runs on population metrics (sciplex_interp.chemcpa_eval): every training
# condition and the held-out targets of $CHEMCPA_TASK_SETS, generated from permitted controls
# (CPU forward passes). The (variant, seed) runs of $VARIANTS x $SEEDS are dealt round-robin over
# the array tasks, so a few tasks cover many runs without crowding the per-user submit limit.
# Up-to-date outputs are skipped.
set -euo pipefail
REPO_DIR="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/vc_wrf}"
[[ -f "$REPO_DIR/slurm/interp/preamble.sh" ]] || REPO_DIR=/zfsauton2/home/yixiz/vc_wrf
source "$REPO_DIR/slurm/interp/preamble.sh" "$@"

read -ra variants <<<"$VARIANTS"
read -ra seeds <<<"$SEEDS"
read -ra task_sets <<<"$CHEMCPA_TASK_SETS"
read -ra shift <<<"$SHIFT_PREDICTORS"
n=$(( ${#variants[@]} * ${#seeds[@]} ))
for (( i = SLURM_ARRAY_TASK_ID; i < n; i += SLURM_ARRAY_TASK_COUNT )); do
  variant="${variants[i / ${#seeds[@]}]}"
  seed="${seeds[i % ${#seeds[@]}]}"
  echo "== run $i: VARIANT=$variant SEED=$seed"
  python -m sciplex_interp.chemcpa_eval --variant "$variant" --seed "$seed" --task-sets "${task_sets[@]}" \
    --shift-predictors "${shift[@]}" --data "$PANEL" --root "$INTERP_ROOT"
done
