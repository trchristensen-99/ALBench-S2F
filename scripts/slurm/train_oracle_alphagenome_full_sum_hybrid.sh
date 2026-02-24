#!/bin/bash
# AlphaGenome boda-sum head with full shift augmentation (hybrid: 50% cache, 50% encoder+shift).
# Reuses outputs/ag_flatten/embedding_cache. Output: outputs/ag_sum_hybrid.
#SBATCH --job-name=ag_sum_hybrid
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpuq
#SBATCH --time=12:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1

source /etc/profile.d/modules.sh
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD:$PYTHONPATH"

# Hybrid runs encoder for 50% of batches; use smaller batch to avoid GPU OOM.
uv run python experiments/train_oracle_alphagenome_full.py \
    ++head_arch="boda-sum-512-512" \
    ++aug_mode="hybrid" \
    ++batch_size=64 \
    ++gpu=0 \
    ++seed=42 \
    ++output_dir=outputs/ag_sum_hybrid \
    ++cache_dir=outputs/ag_flatten/embedding_cache
