#!/bin/bash
#SBATCH --job-name=std-doloris
#SBATCH --partition=legacy
#SBATCH --gres=gpu:rtx_2080_ti:1
#SBATCH --exclude=gpu15
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=06:00:00
# One array task per group of $FIT_GROUPS: for each seed, train DOLORIS (mask model, then diffusion;
# resumes from the last checkpoint) and generate val/test/ood cells under both rules (resumable).
set -euo pipefail
REPO_DIR="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/vc_wrf}"
[[ -f "$REPO_DIR/slurm/standard/preamble.sh" ]] || REPO_DIR=/zfsauton2/home/yixiz/vc_wrf
source "$REPO_DIR/slurm/standard/preamble.sh" "$@"
GROUP=$(array_item FIT_GROUPS)
IFS=, read -ra group <<<"$GROUP"
for SEED in "${group[@]}"; do
  echo "== doloris seed $SEED"
  python -m sciplex_standard.doloris train --seed "$SEED" --file "$STD_FILE" --root "$STD_ROOT"
  python -m sciplex_standard.doloris generate --seed "$SEED" --file "$STD_FILE" --root "$STD_ROOT"
done
