# Configuration for the DOLORIS reproduction; sourced by submit.sh and preamble.sh.
#
# Arguments: a file path is sourced as an env file (submit.sh writes one per submission and
# passes it to every job), and VAR=value is exported. Defaults fill in anything still unset.

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
# Uni-Mol (unimol_tools 0.1.6, CPU torch) in its own environment; weights pre-downloaded from
# huggingface dptech/Uni-Mol-Models (mol_pre_all_h_220816.pt, mol.dict.txt).
export UNIMOL_VENV="${UNIMOL_VENV:-/zfsauton/scratch/yixiz/envs/unimol}"
export UNIMOL_WEIGHTS="${UNIMOL_WEIGHTS:-/zfsauton/scratch/yixiz/envs/unimol_weights}"
# For the parity check only: the released code (github.com/ChangxiChi/Doloris @ 1774472) and a
# directory holding torch_geometric 2.6.1 (+ psutil), added to the path of that job alone.
export DOLORIS_SRC="${DOLORIS_SRC:-/zfsauton/scratch/yixiz/envs/doloris_src_dir/doloris_src}"
export PYG_SITE="${PYG_SITE:-/zfsauton/scratch/yixiz/envs/pyg261}"
export DATA_DIR="${DATA_DIR:-/zfsauton/scratch/yixiz/sciplex3}"
export PANEL="${PANEL:-$DATA_DIR/processed/a549_hdaci_v1.h5ad}"
export DOLORIS_ROOT="${DOLORIS_ROOT:-$REPO/results/doloris}"
# configs/doloris_<CONFIG_NAME>.json; runs go to $DOLORIS_ROOT/runs/<name>, methods are doloris_<name>_<rule>.
export CONFIG_NAME="${CONFIG_NAME:-v1}"
export SEEDS="${SEEDS:-0 1 2}"
# Seeds trained in one GPU job, comma-separated, groups separated by spaces (the debug QoS allows
# 2 submitted jobs per user). Default: all seeds in one job.
export FIT_GROUPS="${FIT_GROUPS:-$(tr ' ' ',' <<<"$SEEDS")}"
export SPLITS="${SPLITS:-val test}"
# Drug -> SMILES as chemCPA recorded it from the GEO chemical metadata.
export SMILES_RUN="${SMILES_RUN:-$REPO/results/chemcpa/gauss_v1/seed_0/run.json}"
export STAGES="${STAGES:-tests parity embed fit score report}"
# Job id(s), colon-separated, that every stage of this submission also waits for (afterok).
export AFTER="${AFTER:-}"
