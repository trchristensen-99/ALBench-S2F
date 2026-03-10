#!/bin/bash
#SBATCH --job-name=ag_sum_compact
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpuq
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1

set -e
source /etc/profile.d/modules.sh
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD:$PYTHONPATH"
source scripts/slurm/setup_hpc_deps.sh

# 384-bp compact window, no_shift (reuses shared cache built by build_compact_cache.sh).
# T=3 tokens (vs T=5 for 600bp); no N-padding; shifts redistribute real flank sequence.
uv run python experiments/train_oracle_alphagenome_full.py \
    ++head_arch="boda-sum-512-512" \
    ++aug_mode="no_shift" \
    ++use_compact_window=true \
    ++compact_window_bp=384 \
    ++gpu=0 \
    ++seed=42 \
    ++output_dir=outputs/ag_sum_compact \
    ++cache_dir=outputs/ag_compact/embedding_cache_compact
