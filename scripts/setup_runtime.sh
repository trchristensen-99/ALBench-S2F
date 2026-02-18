#!/bin/bash
set -euo pipefail

# One-command runtime bootstrap:
# 1) sync deps
# 2) install compatible torch wheel for this host
# 3) display wandb auth status

uv sync --extra dev
uv run python scripts/auto_configure_torch.py --apply
uv run wandb status
