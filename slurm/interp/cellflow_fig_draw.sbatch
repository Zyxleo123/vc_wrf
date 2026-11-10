#!/bin/bash
#SBATCH --job-name=interp-cellflow-fig-draw
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:15:00
# PNGs of $FIG_ARM on $FIG_SPLIT from the files cellflow_fig_collect saved; redraws without the panel.
set -euo pipefail
REPO_DIR="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/vc_wrf}"
[[ -f "$REPO_DIR/slurm/interp/preamble.sh" ]] || REPO_DIR=/zfsauton2/home/yixiz/vc_wrf
source "$REPO_DIR/slurm/interp/preamble.sh" "$@"

python -m sciplex_interp.cellflow_figures draw --arm "${FIG_ARM:?set FIG_ARM}" --split "$FIG_SPLIT" --root "$INTERP_ROOT"
