#!/bin/bash
#SBATCH --job-name=interp-cellflow-tests
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
# Unit tests of the CellFlow study code, on a tiny synthetic panel with CPU JAX.
set -euo pipefail
REPO_DIR="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/vc_wrf}"
[[ -f "$REPO_DIR/slurm/interp/preamble.sh" ]] || REPO_DIR=/zfsauton2/home/yixiz/vc_wrf
JOB_VENV_VAR=CELLFLOW_VENV
source "$REPO_DIR/slurm/interp/preamble.sh" "$@"

JAX_PLATFORMS=cpu python -m pytest tests/test_cellflow.py -q -x
