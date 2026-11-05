# Configuration for the WRF analysis pipeline; sourced by submit.sh and preamble.sh.
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
# Everything the analysis writes: extracts/, tables/, figures/, report.md.
export ANALYSIS_ROOT="${ANALYSIS_ROOT:-$REPO/results/analysis}"
# Stages to submit, in order; each waits (afterok) for the previous submitted one.
export STAGES="${STAGES:-measured extract analyze figures}"
# Bootstrap resamples for every paired comparison.
export N_BOOT="${N_BOOT:-10000}"
# Sampling seeds per adjacent-dose-mixture ablation (stages adjacent, adjacent_compare).
export AUDIT_SAMPLING_SEEDS="${AUDIT_SAMPLING_SEEDS:-20}"
