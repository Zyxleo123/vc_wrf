#!/bin/bash
#SBATCH --job-name=wrfan-measured
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
# Measured-cell extract: r(x) of every panel cell, held-out gene means/SDs, and (once)
# the non-program PCA basis in artifacts/analysis_v1/ (control + train cells only).
set -euo pipefail
REPO_DIR="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/vc_wrf}"
[[ -f "$REPO_DIR/slurm/analysis/preamble.sh" ]] || REPO_DIR=/zfsauton2/home/yixiz/vc_wrf
source "$REPO_DIR/slurm/analysis/preamble.sh" "$@"

python -m sciplex_analysis.extract measured --data "$PANEL" --root "$ANALYSIS_ROOT"
