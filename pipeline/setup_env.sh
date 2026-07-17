#!/usr/bin/env bash
# setup_env.sh -- creates (or updates) the "llmesh" conda environment this
# pipeline needs (meshio/scipy/numpy -- the system Python here has no
# working pip and lacks all three). Requires conda already available on
# PATH; if you don't have conda, install Miniconda first:
#   https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
#
# Usage:
#   bash setup_env.sh
#   conda activate llmesh
#   python3 driver.py plan --case-config case_config.json ...
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found on PATH -- install Miniconda first (see comment at the top of this script)" >&2
  exit 1
fi

if conda env list | grep -q '^llmesh '; then
  echo "==> llmesh env already exists, updating it"
  conda env update -n llmesh -f "${SCRIPT_DIR}/environment.yml" --prune
else
  echo "==> creating llmesh env"
  conda env create -f "${SCRIPT_DIR}/environment.yml"
fi

echo "==> done. Activate with: conda activate llmesh"
