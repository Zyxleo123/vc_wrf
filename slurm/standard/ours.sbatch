#!/bin/bash
#SBATCH --job-name=std-ours
#SBATCH --partition=debug
#SBATCH --gres=gpu:rtx_2080_ti:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
# One array task per group of $FIT_GROUPS: for each seed, train Ours (CellFlow environment; a partial
# run restarts) and generate val/test/ood cells under both checkpoint rules (skips finished files).
# debug/rtx_2080_ti: AVAILABLE in myfree (2026-11-10); legacy has room for one job only (user limit 50 submitted). A seed takes ~25-30 min.
set -euo pipefail
REPO_DIR="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/vc_wrf}"
[[ -f "$REPO_DIR/slurm/standard/preamble.sh" ]] || REPO_DIR=/zfsauton2/home/yixiz/vc_wrf
JOB_VENV_VAR=CELLFLOW_VENV
source "$REPO_DIR/slurm/standard/preamble.sh" "$@"
GROUP=$(array_item FIT_GROUPS)
IFS=, read -ra group <<<"$GROUP"
for SEED in "${group[@]}"; do
  echo "== ours seed $SEED"
  python -m sciplex_standard.ours train --seed "$SEED" --file "$STD_FILE" --root "$STD_ROOT"
  python -m sciplex_standard.ours generate --seed "$SEED" --file "$STD_FILE" --root "$STD_ROOT"
done
