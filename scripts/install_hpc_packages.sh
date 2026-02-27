#!/bin/bash
# One-time setup: install all HPC-specific packages into the uv venv.
# Run this script ONCE from the CSHL login node (bamdev4) before submitting jobs.
# All subsequent SLURM jobs use these pre-installed packages via --no-sync.
#
# Usage: cd /grid/wsbs/home_norepl/christen/ALBench-S2F && bash scripts/install_hpc_packages.sh
#
# After running, verify with:
#   .venv/bin/python -c "import alphagenome_research, alphagenome_ft, anndata, h5py; print('OK')"

set -e
cd "$(dirname "$0")/.." || exit 1

AG_FT_PATH="${HOME}/alphagenome_ft-main"
AG_RES_REV="35ea7aa5"

echo "=== Step 0: uv sync (rebuild venv from pyproject.toml) ==="
uv sync

echo "=== Step 1: GPU-capable JAX (CUDA 12) ==="
UV_LINK_MODE=copy uv pip install "jax[cuda12]"

echo "=== Step 2: alphagenome base SDK ==="
UV_LINK_MODE=copy uv pip install "alphagenome==0.6.0"

echo "=== Step 3: alphagenome_ft (local) ==="
if [ -d "$AG_FT_PATH" ]; then
  UV_LINK_MODE=copy uv pip install "$AG_FT_PATH"
else
  echo "ERROR: $AG_FT_PATH not found. Aborting." >&2
  exit 1
fi

echo "=== Step 4: alphagenome_research (from GitHub) ==="
UV_LINK_MODE=copy uv pip install \
  "alphagenome-research @ git+https://github.com/google-deepmind/alphagenome_research@${AG_RES_REV}"

echo "=== Step 5: Misc JAX / scientific packages ==="
UV_LINK_MODE=copy uv pip install \
  "jmp==0.0.4" \
  jaxtyping \
  dm-haiku \
  chex \
  orbax-checkpoint \
  anndata \
  h5py \
  pytz

echo "=== Verification ==="
.venv/bin/python -c "
import alphagenome_research
import alphagenome_ft
import alphagenome
import anndata
import h5py
import jmp
import haiku
import chex
import orbax.checkpoint
import pytz
print('ALL OK — venv ready for SLURM jobs.')
"
