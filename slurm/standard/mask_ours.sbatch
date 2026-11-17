#!/bin/bash
#SBATCH --job-name=std-mask-ours
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
# Post-hoc: drop DOLORIS's sparsity mask (the trained mask model + nearest-condition zero pattern)
# onto Ours' saved cells, writing preds/{ours_masked,ours_final_masked}/seed_*/<split>.h5ad. CPU only:
# it reads the saved Ours cells and the trained mask models and runs the mask GAT forward. Needs the
# DOLORIS runs (mask_model.pt) and the Ours preds already on disk; skips masked files that exist.
set -euo pipefail
REPO_DIR="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/vc_wrf}"
[[ -f "$REPO_DIR/slurm/standard/preamble.sh" ]] || REPO_DIR=/zfsauton2/home/yixiz/vc_wrf
source "$REPO_DIR/slurm/standard/preamble.sh" "$@"
python -m sciplex_standard.mask_ours --root "$STD_ROOT" --file "$STD_FILE" \
  --seeds ${SEEDS} --splits ${SPLITS} --device cpu
