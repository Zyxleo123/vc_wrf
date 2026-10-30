#!/bin/bash
#SBATCH --job-name=chemcpa-evaluate
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
# One array task per (variant, seed): generate held-out-dose cells from permitted
# controls (a small MLP forward pass, so CPU) and score them with the frozen spec.
# Skips finished evaluations on resubmit.
set -euo pipefail
REPO_DIR="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/vc_wrf}"
[[ -f "$REPO_DIR/slurm/chemcpa/preamble.sh" ]] || REPO_DIR=/zfsauton2/home/yixiz/vc_wrf
source "$REPO_DIR/slurm/chemcpa/preamble.sh" "$@"
array_task

IFS=, read -ra evals <<<"$EVALS"
for item in "${evals[@]}"; do
  read -r split offset <<<"$item"
  python -m sciplex_baselines.chemcpa evaluate --data "$PANEL" --out "$RUN_ROOT/$VARIANT" \
    --seeds "$SEED" --split "$split" --sampling-seed "$offset" --device cpu --threads "$THREADS"
done
