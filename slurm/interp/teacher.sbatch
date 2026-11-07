#!/bin/bash
#SBATCH --job-name=interp-teacher
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
# Fit the mean-level dose-response teacher (MLP auto-decoder) on the training means and write its
# prediction for every condition from labels alone, for the CellFlow jobs (sciplex_interp.teacher).
set -euo pipefail
REPO_DIR="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/vc_wrf}"
[[ -f "$REPO_DIR/slurm/interp/preamble.sh" ]] || REPO_DIR=/zfsauton2/home/yixiz/vc_wrf
source "$REPO_DIR/slurm/interp/preamble.sh" "$@"

for components in $TEACHER_COMPONENTS; do
  name="autodec_k${components}"
  [[ "$components" == 20 ]] && name=autodec
  python -m sciplex_interp.teacher --components "$components" --name "$name" --data "$PANEL" --root "$INTERP_ROOT"
done
