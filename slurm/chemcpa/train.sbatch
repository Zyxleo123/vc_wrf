#!/bin/bash
#SBATCH --job-name=chemcpa-train
#SBATCH --partition=legacy
#SBATCH --gres=gpu:rtx_2080_ti:1
#SBATCH --exclude=gpu15
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
# One array task per (variant, seed): train chemCPA on train + control cells and keep
# the checkpoint with the lowest validation MMD². Skips a finished seed on resubmit.
# gpu15 is excluded: three tasks hung there in CUDA start-up (jobs 91136-91138, 2026-10-30).
set -euo pipefail
REPO_DIR="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/vc_wrf}"
[[ -f "$REPO_DIR/slurm/chemcpa/preamble.sh" ]] || REPO_DIR=/zfsauton2/home/yixiz/vc_wrf
source "$REPO_DIR/slurm/chemcpa/preamble.sh" "$@"
array_task

python -m sciplex_baselines.chemcpa train --data "$PANEL" --chemicals "$CHEMICALS" \
  --config "configs/chemcpa_$VARIANT.json" --out "$RUN_ROOT/$VARIANT" \
  --seeds "$SEED" --device cuda --threads "$THREADS"
