#!/bin/bash
#SBATCH --job-name=interp-cellflow-test
#SBATCH --partition=legacy
#SBATCH --gres=gpu:rtx_2080_ti:1
#SBATCH --exclude=gpu15
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
# One array task per (variant, seed): generate from the selected checkpoint and score with the frozen
# evaluator, for every "<split> <offset>" in $EVALS. Generation needs the model, so this runs on a
# GPU; comparison tables read the saved rows on the CPU partition. Finished evaluations are skipped.
set -euo pipefail
REPO_DIR="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/vc_wrf}"
[[ -f "$REPO_DIR/slurm/interp/preamble.sh" ]] || REPO_DIR=/zfsauton2/home/yixiz/vc_wrf
JOB_VENV_VAR=CELLFLOW_VENV
source "$REPO_DIR/slurm/interp/preamble.sh" "$@"
array_task

IFS=, read -ra evals <<<"$EVALS"
for item in "${evals[@]}"; do
  read -r split offset <<<"$item"
  python -m sciplex_interp.cellflow evaluate --variant "$VARIANT" --seed "$SEED" --split "$split" \
    --sampling-seed "$offset" --data "$PANEL" --root "$INTERP_ROOT"
done
