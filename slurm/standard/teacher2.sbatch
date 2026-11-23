#!/bin/bash
#SBATCH --job-name=std-teacher2
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=02:00:00
# Condition response embedding v2: analogue retrieval + a separately calibrated magnitude.
set -euo pipefail
REPO_DIR="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/vc_wrf}"
[[ -f "$REPO_DIR/slurm/standard/preamble.sh" ]] || REPO_DIR=/zfsauton2/home/yixiz/vc_wrf
source "$REPO_DIR/slurm/standard/preamble.sh" "$@"
python -m sciplex_standard.teacher2 --file "$STD_FILE" --root "$STD_ROOT" \
  --molecules "$STD_ROOT/unimol_embed.npz" --smiles "$STD_ROOT/smiles.csv" \
  --rep "${T2_REP:-rdkit_morgan}" --out "$STD_ROOT/teacher/${T2_FILE:-std_autodec_v2.npz}"
