#!/bin/bash
#SBATCH --job-name=interp-teacher-viz-collect
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=01:00:00
# The teacher's vectors for every condition, what each coordinate means, and whether close conditions are close.
# Reads the saved teacher, the panel's training and validation cells, SMILES and gene sets; no test measurement.
set -euo pipefail
REPO_DIR="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/vc_wrf}"
[[ -f "$REPO_DIR/slurm/interp/preamble.sh" ]] || REPO_DIR=/zfsauton2/home/yixiz/vc_wrf
source "$REPO_DIR/slurm/interp/preamble.sh" "$@"

python -m sciplex_interp.teacher_viz collect --data "$PANEL" --root "$INTERP_ROOT" --chemicals "$CHEMICALS"
