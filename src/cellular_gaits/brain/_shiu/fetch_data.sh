#!/usr/bin/env bash
# Fetch the heavy v783 connectome data for the vendored Shiu brain model.
#
# Only the ~97 MB Connectivity_783.parquet is gitignored and must be fetched;
# the 3.2 MB Completeness_783.csv is committed alongside this script. Both files
# are pulled verbatim from the public upstream repo (MIT-licensed):
#   https://github.com/philshiu/Drosophila_brain_model
#
# Usage:  bash fetch_data.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$HERE/data"
RAW="https://raw.githubusercontent.com/philshiu/Drosophila_brain_model/main"

mkdir -p "$DATA_DIR"

fetch() {
  local name="$1"
  local dest="$DATA_DIR/$name"
  if [[ -f "$dest" ]]; then
    echo ">>> $name already present ($(du -h "$dest" | cut -f1)); skipping."
    return
  fi
  echo ">>> Downloading $name ..."
  curl -fL --retry 3 -o "$dest" "$RAW/$name"
  echo "    saved $(du -h "$dest" | cut -f1) to $dest"
}

# The big one (gitignored):
fetch "Connectivity_783.parquet"
# Committed already, but fetch if a fresh checkout somehow lacks it:
fetch "Completeness_783.csv"

echo ">>> Done. v783 data ready in $DATA_DIR"
