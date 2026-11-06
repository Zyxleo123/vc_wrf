#!/bin/bash
#SBATCH --job-name=interp-dose-draw
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:15:00
# Dose figures from the files dose_collect saved; redraws without the panel or any model.
set -euo pipefail
REPO_DIR="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/vc_wrf}"
[[ -f "$REPO_DIR/slurm/interp/preamble.sh" ]] || REPO_DIR=/zfsauton2/home/yixiz/vc_wrf
source "$REPO_DIR/slurm/interp/preamble.sh" "$@"

python -m sciplex_interp.dose_sweep draw --arm "${FIG_ARM:?set FIG_ARM}" --root "$INTERP_ROOT"
