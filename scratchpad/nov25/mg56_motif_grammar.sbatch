#!/bin/bash
#SBATCH --job-name=mg56
#SBATCH --partition=gpuq
#SBATCH --qos=fast
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=72G
#SBATCH --time=3:50:00
#SBATCH --output=/grid/koo/home/christen/nov25_extra/logs/mg56_%j.log
set -euo pipefail
cd /grid/wsbs/home_norepl/christen/ALBench-S2F

# /grid/wsbs ran out of INODES on 2026-11-25 (52,428,800 of 52,428,800 used), so nothing
# this job produces can land next to the code or the cells it reads. Everything written
# goes to koo (96% of inodes used, 2.5M free) or to node-local /tmp.
EXTRA=/grid/koo/home/christen/nov25_extra
mkdir -p "$EXTRA/logs" "$EXTRA/sidecars" /tmp/$USER/{uv,mpl,tmp}
export ALBENCH_EXTRA_BATTERY="$EXTRA"
export ALBENCH_SIDECAR_ROOT="$EXTRA/sidecars"
export UV_CACHE_DIR=/tmp/$USER/uv MPLCONFIGDIR=/tmp/$USER/mpl TMPDIR=/tmp/$USER/tmp
export HP_ALLOW_UNSTAMPED=1
export AG_S2_ORACLE_DIR=/grid/wsbs/home_norepl/christen/ALBench-S2F/outputs/oracle_v3/s2

# 1. build + oracle-label the grammar-varying motif eval set
.venv/bin/python scratchpad/nov25/build_motif_grammar_evalset.py --n 24000 --out "$EXTRA"

# 2. score every tuned per-size cell on it, and on the existing one-configuration
#    motif_planted set, which was never part of the trained-in battery either
.venv/bin/python scratchpad/nov25/rescore_extra_sets.py \
    --sets motif_grammar motif_planted --max-members 3
