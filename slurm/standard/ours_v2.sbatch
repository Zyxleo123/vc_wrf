#!/bin/bash
#SBATCH --job-name=std-ours-v2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
# One array task per group of $FIT_GROUPS: train Ours on the v2 condition response embedding and generate.
set -euo pipefail
REPO_DIR="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/vc_wrf}"
[[ -f "$REPO_DIR/slurm/standard/preamble.sh" ]] || REPO_DIR=/zfsauton2/home/yixiz/vc_wrf
JOB_VENV_VAR=CELLFLOW_VENV
source "$REPO_DIR/slurm/standard/preamble.sh" "$@"
GROUP=$(array_item FIT_GROUPS)
IFS=, read -ra group <<<"$GROUP"
for SEED in "${group[@]}"; do
  echo "== ${OURS_TAG:-ours_v2} seed $SEED teacher ${T2_FILE:-std_autodec_v2.npz}"
  python -m sciplex_standard.ours train --seed "$SEED" --file "$STD_FILE" --root "$STD_ROOT" \
    --tag "${OURS_TAG:-ours_v2}" --teacher "${T2_FILE:-std_autodec_v2.npz}"
  python -m sciplex_standard.ours generate --seed "$SEED" --file "$STD_FILE" --root "$STD_ROOT" \
    --tag "${OURS_TAG:-ours_v2}" --teacher "${T2_FILE:-std_autodec_v2.npz}"
done
