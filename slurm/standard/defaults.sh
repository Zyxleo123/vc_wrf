# Configuration for the standard sci-Plex3 benchmark (chemCPA data, DOLORIS protocol); sourced by
# submit.sh and preamble.sh. Arguments: an env-file path is sourced, VAR=value is exported.

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
export CELLFLOW_VENV="${CELLFLOW_VENV:-/zfsauton/scratch/yixiz/envs/cellflow}"
export UNIMOL_VENV="${UNIMOL_VENV:-/zfsauton/scratch/yixiz/envs/unimol}"
export UNIMOL_WEIGHTS="${UNIMOL_WEIGHTS:-/zfsauton/scratch/yixiz/envs/unimol_weights}"
# chemCPA's processed sci-Plex3 (sha256 57800e7c...), downloaded from the chemCPA-datasets bucket.
export STD_FILE="${STD_FILE:-/zfsauton/scratch/yixiz/sciplex3/chemcpa/sciplex_complete_middle_subset.h5ad}"
export STD_ROOT="${STD_ROOT:-$REPO/results/standard}"
export SEEDS="${SEEDS:-0 1 2}"
# Seeds run in one GPU job, comma-separated, groups separated by spaces (one seed per legacy job by default).
export FIT_GROUPS="${FIT_GROUPS:-$SEEDS}"
export SPLITS="${SPLITS:-test ood val}"
export STAGES="${STAGES:-prep embed teacher ours doloris score report}"
export AFTER="${AFTER:-}"
