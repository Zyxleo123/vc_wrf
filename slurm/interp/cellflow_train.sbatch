#!/bin/bash
#SBATCH --job-name=interp-cellflow-train
#SBATCH --partition=legacy
#SBATCH --gres=gpu:rtx_2080_ti:1
#SBATCH --exclude=gpu15
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=03:00:00
# One array task per (variant, seed) of $VARIANTS x $SEEDS: train CellFlow from
# configs/interp/cellflow_<variant>.json into $INTERP_ROOT/cellflow/<variant> (min validation MMD²
# checkpoint). Skips a finished seed. legacy/rtx_2080_ti per myfree (2026-11-05); the model has
# about 6M parameters. gpu15 hung in CUDA start-up before (91136-91138).
set -euo pipefail
REPO_DIR="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/vc_wrf}"
[[ -f "$REPO_DIR/slurm/interp/preamble.sh" ]] || REPO_DIR=/zfsauton2/home/yixiz/vc_wrf
JOB_VENV_VAR=CELLFLOW_VENV
source "$REPO_DIR/slurm/interp/preamble.sh" "$@"
array_task

extra=()
[[ -n "$CELLFLOW_CONFIG" ]] && extra=(--config "$CELLFLOW_CONFIG")
python -m sciplex_interp.cellflow train --variant "$VARIANT" --seed "$SEED" --data "$PANEL" --root "$INTERP_ROOT" "${extra[@]}"
