#!/bin/bash
#SBATCH --job-name=std-calib
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
# Marginal calibration of a method's generated cells, scored over a grid of gamma.
set -euo pipefail
REPO_DIR="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/vc_wrf}"
[[ -f "$REPO_DIR/slurm/standard/preamble.sh" ]] || REPO_DIR=/zfsauton2/home/yixiz/vc_wrf
source "$REPO_DIR/slurm/standard/preamble.sh" "$@"
python -m sciplex_standard.calibrate --file "$STD_FILE" --root "$STD_ROOT" \
  --method "${CAL_METHOD:-ours}" --splits ${CAL_SPLITS:-val ood} --seeds ${SEEDS} \
  --gammas ${CAL_GAMMAS:-0.0 0.25 0.5 0.75 0.9 1.0} --zero-rate ${CAL_ZERO:-vehicle} ${CAL_CENTRE:+--centre} ${CAL_WRITE:+--write-gamma $CAL_WRITE}
