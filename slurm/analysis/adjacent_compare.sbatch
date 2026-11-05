#!/bin/bash
#SBATCH --job-name=wrfan-adjacent-compare
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
# Paired condition bootstrap of every adjacent-dose ablation against every model, from the
# saved rows of adjacent.sbatch and results/analysis/tables/per_condition.csv only.
set -euo pipefail
REPO_DIR="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/vc_wrf}"
[[ -f "$REPO_DIR/slurm/analysis/preamble.sh" ]] || REPO_DIR=/zfsauton2/home/yixiz/vc_wrf
source "$REPO_DIR/slurm/analysis/preamble.sh" "$@"

python -m sciplex_analysis.adjacent_audit compare --root "$ANALYSIS_ROOT" --n-boot "$N_BOOT"
