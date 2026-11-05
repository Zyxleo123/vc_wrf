#!/bin/bash
#SBATCH --job-name=interp-chemcpa-compare
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
# Paired comparison of the chosen chemCPA variants ($VARIANTS) with original chemCPA, CellFlow and
# the reference rows, from frozen per-condition rows only (sciplex_interp.chemcpa_compare).
set -euo pipefail
REPO_DIR="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/vc_wrf}"
[[ -f "$REPO_DIR/slurm/interp/preamble.sh" ]] || REPO_DIR=/zfsauton2/home/yixiz/vc_wrf
source "$REPO_DIR/slurm/interp/preamble.sh" "$@"

read -ra variants <<<"$VARIANTS"
read -ra primary <<<"$COMPARE_PRIMARY"
python -m sciplex_interp.chemcpa_compare --variants "${variants[@]}" --primary "${primary[@]}" \
  --out "$REPO/docs/interp"
