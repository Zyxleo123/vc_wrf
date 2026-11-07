#!/bin/bash
#SBATCH --job-name=interp-cellflow-fig-umap
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
# Joint UMAP figure of $FIG_ARM on $FIG_SPLIT. Unlike cellflow_fig_draw it reads the panel and refits UMAP (random_state=0).
set -euo pipefail
REPO_DIR="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/vc_wrf}"
[[ -f "$REPO_DIR/slurm/interp/preamble.sh" ]] || REPO_DIR=/zfsauton2/home/yixiz/vc_wrf
source "$REPO_DIR/slurm/interp/preamble.sh" "$@"

python -m sciplex_interp.cellflow_figures umap --arm "${FIG_ARM:?set FIG_ARM}" --split "$FIG_SPLIT" --data "$PANEL" --root "$INTERP_ROOT"
