#!/bin/bash
#SBATCH --job-name=doloris-embed
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
# Uni-Mol CLS embeddings of the 26 drugs (own environment, CPU). Skips if the file exists.
set -euo pipefail
REPO_DIR="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/vc_wrf}"
[[ -f "$REPO_DIR/slurm/doloris/preamble.sh" ]] || REPO_DIR=/zfsauton2/home/yixiz/vc_wrf
JOB_VENV_VAR=UNIMOL_VENV
source "$REPO_DIR/slurm/doloris/preamble.sh" "$@"

python -I sciplex_doloris/unimol_embed.py --run-json "$SMILES_RUN" --weights-dir "$UNIMOL_WEIGHTS" \
  --out "$DOLORIS_ROOT/unimol_embed.npz"
