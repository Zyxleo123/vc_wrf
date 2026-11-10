#!/bin/bash
#SBATCH --job-name=doloris-tests
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:20:00
# Unit tests of the DOLORIS port and the shared metrics (CPU).
set -euo pipefail
REPO_DIR="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/vc_wrf}"
[[ -f "$REPO_DIR/slurm/doloris/preamble.sh" ]] || REPO_DIR=/zfsauton2/home/yixiz/vc_wrf
source "$REPO_DIR/slurm/doloris/preamble.sh" "$@"

python -m pytest -q tests/test_doloris.py tests/test_metrics.py
