#!/bin/bash
#SBATCH --job-name=interp-cellflow-fig-collect
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
# Metric rows and per-cell profiles of $FIG_ARM and the baselines on $FIG_SPLIT, for the figures. Reads saved
# cells only: the arm must already have been scored on that split by cellflow_test.
set -euo pipefail
REPO_DIR="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/vc_wrf}"
[[ -f "$REPO_DIR/slurm/interp/preamble.sh" ]] || REPO_DIR=/zfsauton2/home/yixiz/vc_wrf
source "$REPO_DIR/slurm/interp/preamble.sh" "$@"

python -m sciplex_interp.cellflow_figures collect --arm "${FIG_ARM:?set FIG_ARM}" --split "$FIG_SPLIT" \
  --chemcpa "$FIG_CHEMCPA" --data "$PANEL" --root "$INTERP_ROOT"
