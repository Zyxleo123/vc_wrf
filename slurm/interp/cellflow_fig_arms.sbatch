#!/bin/bash
#SBATCH --job-name=interp-cellflow-fig-arms
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
# Every finished CellFlow arm from saved history.jsonl only: the ranking behind "best arm" and the
# training curves (sciplex_interp.cellflow_figures arms).
set -euo pipefail
REPO_DIR="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/vc_wrf}"
[[ -f "$REPO_DIR/slurm/interp/preamble.sh" ]] || REPO_DIR=/zfsauton2/home/yixiz/vc_wrf
source "$REPO_DIR/slurm/interp/preamble.sh" "$@"

python -m sciplex_interp.cellflow_figures arms --root "$INTERP_ROOT"
