#!/bin/bash
#SBATCH --job-name=std-diagnose
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
# Failure-case diagnosis on the standard benchmark: mean-shift gain/direction per condition,
# magnitude-rescaling oracle, the teacher on unseen drugs, and chemistry-vs-response retrieval.
set -euo pipefail
REPO_DIR="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/vc_wrf}"
[[ -f "$REPO_DIR/slurm/standard/preamble.sh" ]] || REPO_DIR=/zfsauton2/home/yixiz/vc_wrf
source "$REPO_DIR/slurm/standard/preamble.sh" "$@"
python -m sciplex_standard.diagnose --file "$STD_FILE" --root "$STD_ROOT" \
  --molecules "$STD_ROOT/unimol_embed.npz" --splits ${DIAG_SPLITS:-ood test}
