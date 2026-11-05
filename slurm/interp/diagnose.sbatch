#!/bin/bash
#SBATCH --job-name=interp-diagnose
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-1
# Task 0: real-data diagnostics (structure, mean noise floors, interpolation shapes).
# Task 1: chemCPA fit-vs-interpolation decomposition from the existing checkpoints (CPU).
# Inner and validation tasks only.
set -euo pipefail
REPO_DIR="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/vc_wrf}"
[[ -f "$REPO_DIR/slurm/interp/preamble.sh" ]] || REPO_DIR=/zfsauton2/home/yixiz/vc_wrf
source "$REPO_DIR/slurm/interp/preamble.sh" "$@"

what=(real chemcpa)
python -m sciplex_interp.diagnose "${what[$SLURM_ARRAY_TASK_ID]}" --data "$PANEL" --root "$INTERP_ROOT"
