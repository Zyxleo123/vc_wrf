#!/bin/bash
#SBATCH --job-name=interp-chemcpa-train
#SBATCH --partition=legacy
#SBATCH --gres=gpu:rtx_2080_ti:1
#SBATCH --exclude=gpu15
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
# One array task per (variant, seed) of $VARIANTS x $SEEDS: train chemCPA from
# configs/interp/chemcpa_<variant>.json into $CHEMCPA_ROOT/<variant> (min validation MMD²
# checkpoint, as for the baselines). Skips a finished seed. legacy/rtx_2080_ti per myfree
# (2026-11-05); chemCPA needs < 2 GB of GPU memory. gpu15 hung in CUDA start-up before (91136-91138).
set -euo pipefail
REPO_DIR="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/vc_wrf}"
[[ -f "$REPO_DIR/slurm/interp/preamble.sh" ]] || REPO_DIR=/zfsauton2/home/yixiz/vc_wrf
source "$REPO_DIR/slurm/interp/preamble.sh" "$@"
array_task

config="configs/interp/chemcpa_$VARIANT.json"
[[ -f "$config" ]] || { echo "missing $config" >&2; exit 1; }
python -m sciplex_baselines.chemcpa train --data "$PANEL" --chemicals "$CHEMICALS" \
  --config "$config" --out "$CHEMCPA_ROOT/$VARIANT" --seeds "$SEED" --device cuda --threads "$THREADS"
