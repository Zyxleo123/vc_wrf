#!/bin/bash
#SBATCH --job-name=doloris-fit
#SBATCH --partition=debug
#SBATCH --gres=gpu:rtx_2080_ti:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
# One array task per group of $FIT_GROUPS (comma-separated seeds, e.g. "0,1 2"): for each seed in the
# group, train (mask model, then diffusion model; resumes from the last saved step) and generate
# (validation RMSE of every checkpoint, then cells for $SPLITS under both checkpoint rules; skips
# finished files). Seeds are grouped because the debug QoS allows 2 submitted jobs per user.
# debug/rtx_2080_ti: the AVAILABLE nodes in myfree (2026-11-10); the model is ~45M parameters at
# batch 32, well inside 11 GB. A task that hits the time limit is simply resubmitted.
set -euo pipefail
REPO_DIR="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/vc_wrf}"
[[ -f "$REPO_DIR/slurm/doloris/preamble.sh" ]] || REPO_DIR=/zfsauton2/home/yixiz/vc_wrf
source "$REPO_DIR/slurm/doloris/preamble.sh" "$@"
GROUP=$(array_item FIT_GROUPS)
echo "== seeds $GROUP"

read -ra splits <<<"$SPLITS"
IFS=, read -ra group <<<"$GROUP"
for SEED in "${group[@]}"; do
  run="$DOLORIS_ROOT/runs/$CONFIG_NAME/seed_$SEED"
  echo "== train seed $SEED -> $run"
  python -m sciplex_doloris.train --config "configs/doloris_$CONFIG_NAME.json" --seed "$SEED" \
    --molecules "$DOLORIS_ROOT/unimol_embed.npz" --data "$PANEL" --out "$run" --device cuda
  echo "== generate seed $SEED"
  python -m sciplex_doloris.generate --run "$run" --molecules "$DOLORIS_ROOT/unimol_embed.npz" --data "$PANEL" \
    --preds-root "$DOLORIS_ROOT/preds" --name "$CONFIG_NAME" --splits "${splits[@]}" --device cuda
done
