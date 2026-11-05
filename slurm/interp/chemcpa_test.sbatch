#!/bin/bash
#SBATCH --job-name=interp-chemcpa-test
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
# Official evaluation of chemCPA runs with the original pipeline (sciplex_baselines.chemcpa
# evaluate: 1024 cells per condition from permitted controls, frozen evaluator), for every
# "<split> <offset>" in $EVALS. Runs of $VARIANTS x $SEEDS are dealt round-robin over the array
# tasks. A variant with configs/interp/chemcpa_<variant>.json lives in $CHEMCPA_ROOT, any other in
# $BASE_CHEMCPA_ROOT. Finished evaluations are skipped.
set -euo pipefail
REPO_DIR="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/vc_wrf}"
[[ -f "$REPO_DIR/slurm/interp/preamble.sh" ]] || REPO_DIR=/zfsauton2/home/yixiz/vc_wrf
source "$REPO_DIR/slurm/interp/preamble.sh" "$@"

read -ra variants <<<"$VARIANTS"
read -ra seeds <<<"$SEEDS"
IFS=, read -ra evals <<<"$EVALS"
n=$(( ${#variants[@]} * ${#seeds[@]} ))
for (( i = SLURM_ARRAY_TASK_ID; i < n; i += SLURM_ARRAY_TASK_COUNT )); do
  variant="${variants[i / ${#seeds[@]}]}"
  seed="${seeds[i % ${#seeds[@]}]}"
  root="$BASE_CHEMCPA_ROOT"
  [[ -f "configs/interp/chemcpa_$variant.json" ]] && root="$CHEMCPA_ROOT"
  for item in "${evals[@]}"; do
    read -r split offset <<<"$item"
    echo "== run $i: VARIANT=$variant SEED=$seed split=$split offset=$offset root=$root"
    python -m sciplex_baselines.chemcpa evaluate --data "$PANEL" --out "$root/$variant" --seeds "$seed" \
      --split "$split" --sampling-seed "$offset" --device cpu --threads "$THREADS"
  done
done
