#!/bin/bash
#SBATCH --job-name=interp-teacher-viz-tests
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
# The statistics behind the teacher's coordinate figures, on synthetic data with known answers.
set -euo pipefail
REPO_DIR="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/vc_wrf}"
[[ -f "$REPO_DIR/slurm/interp/preamble.sh" ]] || REPO_DIR=/zfsauton2/home/yixiz/vc_wrf
source "$REPO_DIR/slurm/interp/preamble.sh" "$@"

python -m pytest -q tests/test_teacher_viz.py
