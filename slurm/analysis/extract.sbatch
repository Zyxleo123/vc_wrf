#!/bin/bash
#SBATCH --job-name=wrfan-extract
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
# One array task per registered chemCPA method (configs/chemcpa_*.json, sorted): compact
# extracts of every saved prediction file. Skips existing extracts on resubmit.
# CellFlow's predictions are on PSC; its extracts are made there (see extract.py).
set -euo pipefail
REPO_DIR="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/vc_wrf}"
[[ -f "$REPO_DIR/slurm/analysis/preamble.sh" ]] || REPO_DIR=/zfsauton2/home/yixiz/vc_wrf
source "$REPO_DIR/slurm/analysis/preamble.sh" "$@"

python -m sciplex_analysis.extract predictions --model chemcpa --task-index "${SLURM_ARRAY_TASK_ID:?array job}" \
  --data "$PANEL" --root "$ANALYSIS_ROOT"
