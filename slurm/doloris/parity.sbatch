#!/bin/bash
#SBATCH --job-name=doloris-parity
#SBATCH --partition=debug
#SBATCH --gres=gpu:rtx_2080_ti:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
# Parity of the port with the released DOLORIS code (tests/parity_doloris.py). Needs a GPU because
# the released model hard-codes 'cuda'. debug/rtx_2080_ti: the AVAILABLE nodes in myfree (2026-11-10).
# Exits 1 if any check fails; nothing depends on it.
set -euo pipefail
REPO_DIR="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/vc_wrf}"
[[ -f "$REPO_DIR/slurm/doloris/preamble.sh" ]] || REPO_DIR=/zfsauton2/home/yixiz/vc_wrf
source "$REPO_DIR/slurm/doloris/preamble.sh" "$@"

python tests/parity_doloris.py --doloris-src "$DOLORIS_SRC" --pyg-site "$PYG_SITE" --out "$DOLORIS_ROOT/parity.json"
