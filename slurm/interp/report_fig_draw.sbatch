#!/bin/bash
#SBATCH --job-name=interp-report-fig-draw
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:15:00
# Report figures of $FIG_ARM on $FIG_SPLIT: one axis per plot, the restricted baseline set. Reads only the
# files cellflow_fig_collect saved (collect $FIG_SPLIT with FIG_CHEMCPA=gauss_v1 first). Redraws without the panel.
set -euo pipefail
REPO_DIR="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/vc_wrf}"
[[ -f "$REPO_DIR/slurm/interp/preamble.sh" ]] || REPO_DIR=/zfsauton2/home/yixiz/vc_wrf
source "$REPO_DIR/slurm/interp/preamble.sh" "$@"

python -m sciplex_interp.report_figures draw --arm "${FIG_ARM:?set FIG_ARM}" --split "$FIG_SPLIT" --root "$INTERP_ROOT"
# The dose story (validation drugs): reads figures/dose/, drawn from the arm named by FIG_ARM.
python -m sciplex_interp.report_figures dose --arm "$FIG_ARM" --root "$INTERP_ROOT"
