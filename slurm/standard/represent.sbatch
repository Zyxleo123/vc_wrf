#!/bin/bash
#SBATCH --job-name=std-represent
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
# Leave-one-drug-out study of the structure -> response map: molecular representations x predictors.
set -euo pipefail
REPO_DIR="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/vc_wrf}"
[[ -f "$REPO_DIR/slurm/standard/preamble.sh" ]] || REPO_DIR=/zfsauton2/home/yixiz/vc_wrf
source "$REPO_DIR/slurm/standard/preamble.sh" "$@"
python -m sciplex_standard.represent --file "$STD_FILE" --root "$STD_ROOT" \
  --molecules "$STD_ROOT/unimol_embed.npz" --smiles "$STD_ROOT/smiles.csv"
