#!/bin/bash
#SBATCH --job-name=interp-cellflow-compare
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
# Paired comparison of the chosen CellFlow variants ($VARIANTS) with the matched base
# ($CELLFLOW_BASE), published CellFlow, chemCPA + mean consistency and the reference rows, from saved
# per-condition rows only (sciplex_interp.cellflow_compare).
set -euo pipefail
REPO_DIR="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/vc_wrf}"
[[ -f "$REPO_DIR/slurm/interp/preamble.sh" ]] || REPO_DIR=/zfsauton2/home/yixiz/vc_wrf
source "$REPO_DIR/slurm/interp/preamble.sh" "$@"

read -ra variants <<<"$VARIANTS"
read -ra labels <<<"$CELLFLOW_LABELS"
python -m sciplex_interp.cellflow_compare --variants "${variants[@]}" --base "$CELLFLOW_BASE" \
  --labels "${labels[@]}" --root "$INTERP_ROOT" --out "$REPO/docs/interp"
