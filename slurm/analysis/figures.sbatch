#!/bin/bash
#SBATCH --job-name=wrfan-figures
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
# Figures and report.md from the analysis tables and extracts only; redraws in minutes.
set -euo pipefail
REPO_DIR="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/vc_wrf}"
[[ -f "$REPO_DIR/slurm/analysis/preamble.sh" ]] || REPO_DIR=/zfsauton2/home/yixiz/vc_wrf
source "$REPO_DIR/slurm/analysis/preamble.sh" "$@"

python -m sciplex_analysis.figures --root "$ANALYSIS_ROOT"
