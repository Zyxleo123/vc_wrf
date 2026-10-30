# Configuration for the chemCPA SLURM pipeline; sourced by submit.sh and preamble.sh.
#
# Arguments: a file path is sourced as an env file (submit.sh writes one per
# submission and passes it to every job), and VAR=value is exported. Defaults
# fill in anything still unset. No other side effects.

for arg in "$@"; do
  if [[ "$arg" == *=* ]]; then
    export "$arg"
  elif [[ -f "$arg" ]]; then
    set -a
    source "$arg"
    set +a
  else
    echo "defaults.sh: argument is neither VAR=value nor a file: $arg" >&2
    return 2
  fi
done

export REPO="${REPO:-/zfsauton2/home/yixiz/vc_wrf}"
export VENV="${VENV:-$REPO/.venv}"
export DATA_DIR="${DATA_DIR:-/zfsauton/scratch/yixiz/sciplex3}"
export PANEL="${PANEL:-$DATA_DIR/processed/a549_hdaci_v1.h5ad}"
export CHEMICALS="${CHEMICALS:-$DATA_DIR/raw/GSE139944/GSM4150378_sciPlex3_A549_MCF7_K562_hashTable_metadata.txt.gz}"
export RUN_ROOT="${RUN_ROOT:-$REPO/results/chemcpa}"
# Each variant trains configs/chemcpa_<variant>.json into $RUN_ROOT/<variant>.
export VARIANTS="${VARIANTS:-nb_v1 gauss_v1}"
export SEEDS="${SEEDS:-0 1 2}"
# "<split> <sampling offset>" evaluations, separated by commas.
export EVALS="${EVALS:-val 0,test 0,test 1}"
# Cross-variant comparisons go to $RUN_ROOT/<COMPARE_NAME>_<split>_s<offset>.
export COMPARE_NAME="${COMPARE_NAME:-compare}"
# 1: set the processed/ directory aside and rebuild it from the raw GEO download.
export REBUILD_DATA="${REBUILD_DATA:-0}"
# 1: submit without the data stage (the panel must already exist).
export SKIP_DATA="${SKIP_DATA:-0}"
