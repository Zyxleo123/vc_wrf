#!/bin/bash
#SBATCH --job-name=std-score
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
# One array task per split of $SPLITS: DOLORIS's metrics + MMD² for every method's saved cells and the references.
set -euo pipefail
REPO_DIR="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/vc_wrf}"
[[ -f "$REPO_DIR/slurm/standard/preamble.sh" ]] || REPO_DIR=/zfsauton2/home/yixiz/vc_wrf
source "$REPO_DIR/slurm/standard/preamble.sh" "$@"
SPLIT=$(array_item SPLITS)
python -m sciplex_standard.evaluate score --split "$SPLIT" --file "$STD_FILE" --root "$STD_ROOT"
