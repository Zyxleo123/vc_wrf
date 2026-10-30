#!/bin/bash
#SBATCH --job-name=chemcpa-report
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
# Tables and figures from saved scores and generated cells only (no model): one report
# per variant and evaluation, then a cross-variant comparison per evaluation.
set -euo pipefail
REPO_DIR="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/vc_wrf}"
[[ -f "$REPO_DIR/slurm/chemcpa/preamble.sh" ]] || REPO_DIR=/zfsauton2/home/yixiz/vc_wrf
source "$REPO_DIR/slurm/chemcpa/preamble.sh" "$@"

read -ra variants <<<"$VARIANTS"
read -ra seeds <<<"$SEEDS"
IFS=, read -ra evals <<<"$EVALS"
for item in "${evals[@]}"; do
  read -r split offset <<<"$item"
  reports=()
  for variant in "${variants[@]}"; do
    out="$RUN_ROOT/$variant/report_${split}_s${offset}"
    python -m sciplex_baselines.report_chemcpa --data "$PANEL" --runs "$RUN_ROOT/$variant" \
      --split "$split" --sampling-seed "$offset" --seeds "${seeds[@]}" --out "$out"
    reports+=("$out")
  done
  python -m sciplex_baselines.compare_chemcpa --reports "${reports[@]}" --out "$RUN_ROOT/${COMPARE_NAME}_${split}_s${offset}"
done
