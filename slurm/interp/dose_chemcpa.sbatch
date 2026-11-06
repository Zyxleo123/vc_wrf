#!/bin/bash
#SBATCH --job-name=interp-dose-chemcpa
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
# One array task per ($DOSE_CHEMCPA, seed): the same dose sweep for chemCPA, which generates on CPU as its
# evaluation does. Finished sweeps are skipped.
set -euo pipefail
REPO_DIR="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/vc_wrf}"
[[ -f "$REPO_DIR/slurm/interp/preamble.sh" ]] || REPO_DIR=/zfsauton2/home/yixiz/vc_wrf
source "$REPO_DIR/slurm/interp/preamble.sh" "$@"
VARIANTS="$DOSE_CHEMCPA"
array_task

python -m sciplex_interp.dose_sweep generate --family chemcpa --variant "$VARIANT" --seed "$SEED" --data "$PANEL" --root "$INTERP_ROOT"
