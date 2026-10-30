#!/bin/bash
#SBATCH --job-name=chemcpa-data
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
# Build the sci-Plex3 A549 HDACi panel from GEO: verified download, raw-count h5ad,
# panel. build_panel.py writes the panel only if it reproduces the frozen spec and
# split. With REBUILD_DATA=1 the old processed/ is set aside and compared afterwards.
set -euo pipefail
REPO_DIR="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/vc_wrf}"
[[ -f "$REPO_DIR/slurm/chemcpa/preamble.sh" ]] || REPO_DIR=/zfsauton2/home/yixiz/vc_wrf
source "$REPO_DIR/slurm/chemcpa/preamble.sh" "$@"

RAW="$DATA_DIR/raw/GSE139944"
PROCESSED="$DATA_DIR/processed"
if [[ -f "$RAW/SHA256SUMS" ]]; then
  (cd "$RAW" && sha256sum -c SHA256SUMS)
fi
bash scripts/download_sciplex3.sh "$DATA_DIR"

PREVIOUS=""
if [[ "$REBUILD_DATA" == 1 && -d "$PROCESSED" ]]; then
  PREVIOUS="$PROCESSED.prev-$(date +%Y%m%d-%H%M%S)"
  mv "$PROCESSED" "$PREVIOUS"
  echo "set aside $PROCESSED -> $PREVIOUS"
fi
python -I scripts/build_sciplex3_h5ad.py "$DATA_DIR"
if [[ -f "$PANEL" ]]; then
  echo "panel exists, not rebuilding: $PANEL"
else
  python -I scripts/build_panel.py --data-dir "$DATA_DIR"
fi
git diff --exit-code --stat -- artifacts/eval_spec_v1 configs/split_v1.json

if [[ -n "$PREVIOUS" ]]; then
  python -I - "$PREVIOUS/$(basename "$PANEL")" "$PANEL" <<'EOF'
import sys
import anndata as ad
import numpy as np

def dense(m):
    return m.toarray() if hasattr(m, "toarray") else np.asarray(m)

old, new = (ad.read_h5ad(path) for path in sys.argv[1:3])
checks = {
    "obs_names": old.obs_names.equals(new.obs_names),
    "var_names": old.var_names.equals(new.var_names),
    "obs": old.obs.equals(new.obs),
    "var": old.var.equals(new.var),
    "X": np.array_equal(dense(old.X), dense(new.X)),
    "counts": np.array_equal(dense(old.layers["counts"]), dense(new.layers["counts"])),
}
print("rebuilt panel vs previous build:", checks)
print("IDENTICAL" if all(checks.values()) else "DIFFERS (the frozen-spec check above still passed)")
EOF
fi
echo "data stage done: $PANEL"
