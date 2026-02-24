#!/bin/bash
# Reference K562-exact config: boda-flatten-1024-dropout, dropout=0.1, constant LR,
# 100 epochs, patience=5 — matches alphagenome_FT_MPRA/configs/mpra_K562.json exactly.
# Reuses existing 600bp embedding cache from outputs/ag_flatten/embedding_cache.
#SBATCH --job-name=ag_flatten_ref
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=kooq
#SBATCH --qos=koolab
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=200G
#SBATCH --time=24:00:00

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
    ++output_dir=outputs/ag_flatten_ref \
    ++cache_dir=outputs/ag_flatten/embedding_cache \
    ++dropout_rate=0.1 \
    ++lr_schedule=none \
    ++epochs=100
