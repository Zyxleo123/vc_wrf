#!/bin/bash
#SBATCH --job-name=interp-stats
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=02:00:00
#SBATCH --array=0-13
# Mean predictors ("can a simple model infer a condition's mean from the other conditions?"),
# scored on $TASK_SETS. Task groups (one per inner task, one per other task set) are spread
# round-robin over the array; stats_merge.sbatch combines the shards into stats/rows.jsonl and
# stats/means/*.npz. Keep --array in step with STATS_SHARDS in defaults.sh.
set -euo pipefail
REPO_DIR="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/vc_wrf}"
[[ -f "$REPO_DIR/slurm/interp/preamble.sh" ]] || REPO_DIR=/zfsauton2/home/yixiz/vc_wrf
source "$REPO_DIR/slurm/interp/preamble.sh" "$@"

if (( SLURM_ARRAY_TASK_COUNT != STATS_SHARDS )); then
  echo "array has $SLURM_ARRAY_TASK_COUNT tasks but STATS_SHARDS=$STATS_SHARDS" >&2
  exit 1
fi
read -ra task_sets <<<"$TASK_SETS"
python -m sciplex_interp.stats --task-sets "${task_sets[@]}" --shard "$SLURM_ARRAY_TASK_ID" \
  --n-shards "$STATS_SHARDS" --data "$PANEL" --root "$INTERP_ROOT"
