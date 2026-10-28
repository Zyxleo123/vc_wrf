#!/usr/bin/env bash
# Download the sci-Plex3 screen (Srivatsan et al. 2020, GEO GSE139944 / GSM4150378).
# Resumable (curl -C -) and idempotent: a file is skipped once its size matches GEO's
# filelist and it passes `gzip -t`.
#
# usage: scripts/download_sciplex3.sh [DATA_DIR]   (default: /data/user_data/yixiz/sciplex3)
set -euo pipefail

DATA_DIR="${1:-/data/user_data/yixiz/sciplex3}"
RAW="$DATA_DIR/raw/GSE139944"
# FTP is ~4x faster than HTTPS from Babel, and NCBI's HTTPS full-file GET can stall
# indefinitely on the 3 GB matrix; HTTPS stays as a fallback.
SUPPL="ftp.ncbi.nlm.nih.gov/geo/samples/GSM4150nnn/GSM4150378/suppl"
mkdir -p "$RAW"

# name  expected_bytes (from GSE139944 suppl/filelist.txt)
FILES=(
  "GSM4150378_sciPlex3_A549_MCF7_K562_screen_UMI.count.matrix.gz 3096224606"
  "GSM4150378_sciPlex3_A549_MCF7_K562_screen_cell.annotations.txt.gz 2770264"
  "GSM4150378_sciPlex3_A549_MCF7_K562_screen_gene.annotations.txt.gz 806168"
  "GSM4150378_sciPlex3_pData.txt.gz 84673357"
  "GSM4150378_sciPlex3_hashSampleSheet.txt.gz 6441"
  "GSM4150378_sciPlex3_A549_MCF7_K562_hashTable_metadata.txt.gz 64715"
)

for entry in "${FILES[@]}"; do
  read -r name size <<<"$entry"
  out="$RAW/$name"
  if [[ -f "$out.ok" ]]; then echo "skip $name"; continue; fi
  echo "fetch $name ($size bytes)"
  for attempt in 1 2 3 4 5 6; do
    scheme=ftp; (( attempt % 2 == 0 )) && scheme=https
    # abort (and resume on the next attempt) if <100 KB/s for 60 s
    curl -fsSL --speed-limit 100000 --speed-time 60 -C - -o "$out" "$scheme://$SUPPL/$name" && break
    echo "  attempt $attempt ($scheme) failed; retrying" >&2; sleep 10
  done
  got=$(stat -c %s "$out")
  [[ "$got" == "$size" ]] || { echo "size mismatch for $name: $got != $size" >&2; exit 1; }
  gzip -t "$out"
  touch "$out.ok"
done

(cd "$RAW" && sha256sum *.gz > SHA256SUMS)
echo "done: $RAW"
