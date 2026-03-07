#!/bin/bash
# Install Enformer, Borzoi, and Nucleotide Transformer packages on HPC.
# Run ONCE from login node AFTER install_hpc_packages.sh.
#
# Usage: cd /grid/wsbs/home_norepl/christen/ALBench-S2F && bash scripts/install_foundation_models.sh

set -e
cd "$(dirname "$0")/.." || exit 1

echo "=== Step 1: enformer-pytorch (lucidrains) ==="
UV_LINK_MODE=copy uv pip install enformer-pytorch

echo "=== Step 2: borzoi-pytorch (editable) ==="
UV_LINK_MODE=copy uv pip install -e external/borzoi-pytorch/

echo "=== Step 3: nucleotide-transformer (editable, no deps) ==="
UV_LINK_MODE=copy uv pip install --no-deps -e external/nucleotide-transformer/

echo "=== Verification ==="
.venv/bin/python -c "
from enformer_pytorch import Enformer
from borzoi_pytorch import Borzoi
import nucleotide_transformer
print('ALL OK — Enformer, Borzoi, NT installed.')
"
