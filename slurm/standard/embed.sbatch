#!/bin/bash
#SBATCH --job-name=std-embed
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
# Uni-Mol CLS embeddings of the 187 drugs (own environment, CPU). Skips if the file exists.
set -euo pipefail
REPO_DIR="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/vc_wrf}"
[[ -f "$REPO_DIR/slurm/standard/preamble.sh" ]] || REPO_DIR=/zfsauton2/home/yixiz/vc_wrf
JOB_VENV_VAR=UNIMOL_VENV
source "$REPO_DIR/slurm/standard/preamble.sh" "$@"

python -I sciplex_doloris/unimol_embed.py --smiles-csv "$STD_ROOT/smiles.csv" --weights-dir "$UNIMOL_WEIGHTS" \
  --out "$STD_ROOT/unimol_embed.npz"
