#!/bin/bash
#SBATCH --job-name=interp-populations
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
# Population generators built from the adjacent doses' real cells (sciplex_interp.populations),
# with means corrected by the statistics predictors in $POP_PREDICTORS. One array task per
# entry of $TASK_SETS (the submitter sizes --array). Finished (generator, condition, seed) rows
# are kept on resubmission.
set -euo pipefail
REPO_DIR="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/vc_wrf}"
[[ -f "$REPO_DIR/slurm/interp/preamble.sh" ]] || REPO_DIR=/zfsauton2/home/yixiz/vc_wrf
source "$REPO_DIR/slurm/interp/preamble.sh" "$@"

read -ra task_sets <<<"$TASK_SETS"
read -ra predictors <<<"$POP_PREDICTORS"
read -ra kinds <<<"$POP_KINDS"
python -m sciplex_interp.populations --task-sets "${task_sets[@]}" --task-index "$SLURM_ARRAY_TASK_ID" \
  --predictors "${predictors[@]}" --kinds "${kinds[@]}" --seeds "$POP_SEEDS" --data "$PANEL" --root "$INTERP_ROOT"
