#!/bin/bash
#SBATCH --job-name=interp-dose-cellflow
#SBATCH --partition=legacy
#SBATCH --gres=gpu:rtx_2080_ti:1
#SBATCH --exclude=gpu15
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
# One array task per ($DOSE_CELLFLOW, seed): generate the validation drugs over the dose grid from the selected
# checkpoint (sciplex_interp.dose_sweep). Generation needs the model, so this runs on a GPU. Finished sweeps are skipped.
set -euo pipefail
REPO_DIR="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/vc_wrf}"
[[ -f "$REPO_DIR/slurm/interp/preamble.sh" ]] || REPO_DIR=/zfsauton2/home/yixiz/vc_wrf
JOB_VENV_VAR=CELLFLOW_VENV
source "$REPO_DIR/slurm/interp/preamble.sh" "$@"
VARIANTS="$DOSE_CELLFLOW"
array_task

python -m sciplex_interp.dose_sweep generate --family cellflow --variant "$VARIANT" --seed "$SEED" --data "$PANEL" --root "$INTERP_ROOT"
