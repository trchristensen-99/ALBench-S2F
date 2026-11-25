#!/bin/bash
#SBATCH --job-name=mg56
#SBATCH --partition=gpu
#SBATCH --qos=fast
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=72G
#SBATCH --time=3:50:00
#SBATCH --output=/grid/wsbs/home_norepl/christen/ALBench-S2F/scratchpad/nov25/logs/mg56_%j.log
set -euo pipefail
cd /grid/wsbs/home_norepl/christen/ALBench-S2F
export HP_ALLOW_UNSTAMPED=1
export AG_S2_ORACLE_DIR=/grid/wsbs/home_norepl/christen/ALBench-S2F/outputs/oracle_v3/s2

# 1. build + oracle-label the grammar-varying motif eval set
uv run --no-sync python scratchpad/nov25/build_motif_grammar_evalset.py --n 24000

# 2. score every tuned per-size cell on it, and on the existing one-configuration
#    motif_planted set, which was never part of the trained-in battery either
uv run --no-sync python scratchpad/nov25/rescore_extra_sets.py \
    --sets motif_grammar motif_planted --max-members 3
