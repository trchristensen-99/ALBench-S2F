#!/bin/bash
# boda-flatten-1024-dropout, dropout=0.5, constant LR, 100 epochs.
# Same as flatten_ref but with higher dropout (0.5 vs 0.1).
# Reuses existing 600bp embedding cache from outputs/ag_flatten/embedding_cache.
#SBATCH --job-name=ag_flat_do05
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpuq
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00

source /etc/profile.d/modules.sh
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD:$PYTHONPATH"
source scripts/slurm/setup_hpc_deps.sh

uv run python experiments/train_oracle_alphagenome_full.py \
    ++head_arch="boda-flatten-1024-dropout" \
    ++aug_mode="no_shift" \
    ++gpu=0 \
    ++seed=42 \
    ++output_dir=outputs/ag_flatten_ref_do05 \
    ++cache_dir=outputs/ag_flatten/embedding_cache \
    ++dropout_rate=0.5 \
    ++lr_schedule=none \
    ++epochs=100
