#!/bin/bash
#SBATCH --job-name=interp-dose-collect
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
# Dose-response curves, gains and histograms from the saved sweeps and the measured cells of the validation drugs.
set -euo pipefail
REPO_DIR="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/vc_wrf}"
[[ -f "$REPO_DIR/slurm/interp/preamble.sh" ]] || REPO_DIR=/zfsauton2/home/yixiz/vc_wrf
source "$REPO_DIR/slurm/interp/preamble.sh" "$@"

python -m sciplex_interp.dose_sweep collect --arm "${FIG_ARM:?set FIG_ARM}" --data "$PANEL" --root "$INTERP_ROOT"
