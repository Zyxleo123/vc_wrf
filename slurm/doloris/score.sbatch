#!/bin/bash
#SBATCH --job-name=doloris-score
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=01:00:00
# One array task per split of $SPLITS: frozen evaluator on the DOLORIS predictions, then the
# frozen metrics and DOLORIS's metrics for every method from saved cells -> scores/<split>.jsonl.
set -euo pipefail
REPO_DIR="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/vc_wrf}"
[[ -f "$REPO_DIR/slurm/doloris/preamble.sh" ]] || REPO_DIR=/zfsauton2/home/yixiz/vc_wrf
source "$REPO_DIR/slurm/doloris/preamble.sh" "$@"
SPLIT=$(array_item SPLITS)
echo "== split $SPLIT"

python -m sciplex_doloris.score --split "$SPLIT" --root "$DOLORIS_ROOT" --name "$CONFIG_NAME" --data "$PANEL"
