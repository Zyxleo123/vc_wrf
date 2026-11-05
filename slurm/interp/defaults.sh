# Configuration for the dose-interpolation study; sourced by submit.sh and preamble.sh.
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
# CellFlow's pinned JAX environment (experiments/cellflow/requirements.lock.txt at 2c478b8, python 3.12).
export CELLFLOW_VENV="${CELLFLOW_VENV:-/zfsauton/scratch/yixiz/envs/cellflow}"
# Optional config path overriding configs/interp/cellflow_<variant>.json (short test runs).
export CELLFLOW_CONFIG="${CELLFLOW_CONFIG:-}"
# cellflow_compare: the matched baseline and optional variant=label table names (no spaces in a label).
export CELLFLOW_BASE="${CELLFLOW_BASE:-base}"
export CELLFLOW_LABELS="${CELLFLOW_LABELS:-}"
export DATA_DIR="${DATA_DIR:-/zfsauton/scratch/yixiz/sciplex3}"
export PANEL="${PANEL:-$DATA_DIR/processed/a549_hdaci_v1.h5ad}"
export CHEMICALS="${CHEMICALS:-$DATA_DIR/raw/GSE139944/GSM4150378_sciPlex3_A549_MCF7_K562_hashTable_metadata.txt.gz}"
# Everything the study writes (diagnose/, stats/, populations/, report.md).
export INTERP_ROOT="${INTERP_ROOT:-$REPO/results/interp}"
# Stages to submit (see submit.sh for the dependency graph).
export STAGES="${STAGES:-diagnose}"
# Task sets scored by the statistics and population stages: inner and val during
# development; test is added once, for the chosen methods only.
export TASK_SETS="${TASK_SETS:-inner val}"
# chemCPA variants and seeds for the chemcpa stages.
# New chemCPA variants (configs/interp/chemcpa_<variant>.json) train here; the baselines stay in
# results/chemcpa/ (BASE_CHEMCPA_ROOT), kept apart so the WRF analysis registry is unaffected.
export CHEMCPA_ROOT="${CHEMCPA_ROOT:-$INTERP_ROOT/chemcpa}"
export BASE_CHEMCPA_ROOT="${BASE_CHEMCPA_ROOT:-$REPO/results/chemcpa}"
export VARIANTS="${VARIANTS:-}"
export SEEDS="${SEEDS:-0 1 2}"
export EVALS="${EVALS:-val 0}"
# Shards of the stats stage (must match --array in stats.sbatch).
export STATS_SHARDS="${STATS_SHARDS:-14}"
# Held-out task sets scored by chemcpa_eval (val during development; test once at the end), and
# statistics predictors whose means chemcpa_eval also shifts chemCPA's cells onto.
export CHEMCPA_TASK_SETS="${CHEMCPA_TASK_SETS:-val}"
export SHIFT_PREDICTORS="${SHIFT_PREDICTORS:-}"
# Array tasks of chemcpa_eval (runs are dealt round-robin over them).
export EVAL_TASKS="${EVAL_TASKS:-6}"
# Job id(s), colon-separated, that every stage of this submission also waits for (afterok).
export AFTER="${AFTER:-}"
# Population stage: statistics predictors whose means correct the mixture ("a+b" = average of a
# and b), generator kinds (see sciplex_interp/populations.py) and sampling seeds.
export POP_PREDICTORS="${POP_PREDICTORS:-linear linear_w0.3 hill lowrank_r4 traj_l1.5_r0.1_hill autodec_e4_wd0.0001 autodec_e4_wd0.01 autodec_e4_wd0.0001+traj_l1.5_r0.1_hill}"
export POP_KINDS="${POP_KINDS:-mixw shift shiftw shift0 tilt}"
export POP_SEEDS="${POP_SEEDS:-10}"
# chemcpa_compare: "variant=model/method" pairs, each variant's primary reference.
export COMPARE_PRIMARY="${COMPARE_PRIMARY:-gaussmean_l100=chemcpa/gauss_v1 nbmean_l10=chemcpa/nb_v1}"
