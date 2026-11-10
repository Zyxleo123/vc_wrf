#!/bin/bash
#SBATCH --job-name=doloris-report
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:15:00
# Tables from saved score rows only; run alone to redraw.
set -euo pipefail
REPO_DIR="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/vc_wrf}"
[[ -f "$REPO_DIR/slurm/doloris/preamble.sh" ]] || REPO_DIR=/zfsauton2/home/yixiz/vc_wrf
source "$REPO_DIR/slurm/doloris/preamble.sh" "$@"

for split in $SPLITS; do
  python -m sciplex_doloris.report --split "$split" --root "$DOLORIS_ROOT" --name "$CONFIG_NAME"
done
