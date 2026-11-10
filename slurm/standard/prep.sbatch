#!/bin/bash
#SBATCH --job-name=std-prep
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
# The frozen split (once), the evaluation spec (PCA + MMD bandwidths from control + train cells) and
# the drug -> SMILES table for Uni-Mol. Reads chemCPA's file only.
set -euo pipefail
REPO_DIR="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/vc_wrf}"
[[ -f "$REPO_DIR/slurm/standard/preamble.sh" ]] || REPO_DIR=/zfsauton2/home/yixiz/vc_wrf
source "$REPO_DIR/slurm/standard/preamble.sh" "$@"

python -m sciplex_standard.data split --file "$STD_FILE"
python -m sciplex_standard.data spec --file "$STD_FILE"
mkdir -p "$STD_ROOT"
python -m sciplex_standard.data smiles --file "$STD_FILE" --out "$STD_ROOT/smiles.csv"
