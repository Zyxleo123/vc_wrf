#!/bin/bash
#SBATCH --job-name=std-teacher
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
# Ours' condition response embedding for the standard benchmark (training-condition means only).
set -euo pipefail
REPO_DIR="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/vc_wrf}"
[[ -f "$REPO_DIR/slurm/standard/preamble.sh" ]] || REPO_DIR=/zfsauton2/home/yixiz/vc_wrf
source "$REPO_DIR/slurm/standard/preamble.sh" "$@"

python -m sciplex_standard.teacher --molecules "$STD_ROOT/unimol_embed.npz" --file "$STD_FILE" --root "$STD_ROOT"
