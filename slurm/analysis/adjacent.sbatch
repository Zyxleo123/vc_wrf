#!/bin/bash
#SBATCH --job-name=wrfan-adjacent
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --array=0-12
# Adjacent-dose-mixture audit (sciplex_analysis.adjacent_audit build). Task 0 reproduces
# CellFlow's published reference, audits its source cells, scores the program-output chemCPA
# runs on the 13-threshold grid and writes the perfect-predictor floor; task i scores
# ablation i over $AUDIT_SAMPLING_SEEDS sampling seeds. The module refuses to run if this
# --array does not match its task count. Finished row files are skipped on resubmission.
set -euo pipefail
REPO_DIR="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/vc_wrf}"
[[ -f "$REPO_DIR/slurm/analysis/preamble.sh" ]] || REPO_DIR=/zfsauton2/home/yixiz/vc_wrf
source "$REPO_DIR/slurm/analysis/preamble.sh" "$@"

python -m sciplex_analysis.adjacent_audit build --task "$SLURM_ARRAY_TASK_ID" --data "$PANEL" \
  --root "$ANALYSIS_ROOT" --seeds "$AUDIT_SAMPLING_SEEDS"
