#!/bin/bash
set -euo pipefail

# Run commands against the already-configured environment without uv auto-sync.
uv run --no-sync "$@"
