#!/bin/bash
# Submit cross-D LegNet HP-search pipelines (D=5000 and D=100000) using the
# tuned-up FULL standardized_procedure with patience=10, k_parallel=3.
#
# Schedules onto slow_nice to take advantage of overnight idle GPUs.
# Each cell's pipeline takes ~3-6h wall depending on D.

set -euo pipefail
REPO=/grid/wsbs/home_norepl/christen/ALBench-S2F
cd "$REPO"

# slow_nice gives 30-day wall + 20-GPU/user limit; perfect for overnight.
export STAGE_TIME=10:00:00

# D=5000 (cheap; fastest to land — gives an HP basin we can refine later)
echo "[cross-D] launching LegNet D=5000 full pipeline..."
SMOKE=0 ARCH=legnet D_TRAIN=5000 \
    bash scripts/preflight/hpsearch/standardized_procedure.sh

echo "[cross-D] launching LegNet D=100000 full pipeline..."
SMOKE=0 ARCH=legnet D_TRAIN=100000 \
    bash scripts/preflight/hpsearch/standardized_procedure.sh

echo "[cross-D] both pipelines submitted. Each chains Stages 1-5 internally."
