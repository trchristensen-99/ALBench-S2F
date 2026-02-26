#!/bin/bash
#SBATCH --job-name=ag_yeast_flat_hlr
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpuq
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=christen@cshl.edu

source /etc/profile.d/modules.sh
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD:$PYTHONPATH"
mkdir -p logs

# boda-flatten-512-512, lr=1e-3, weight_decay=1e-6, no LR schedule
# Replicates the alphagenome_FT_MPRA reference optimizer (Adam, lr=0.001, wd~=0).
# Shares the existing embedding cache from the Wave-1 runs.
uv run python experiments/train_oracle_alphagenome_yeast.py \
    ++head_arch="boda-flatten-512-512" \
    ++dropout_rate=0.1 \
    ++lr=0.001 \
    ++weight_decay=1e-6 \
    ++lr_schedule=none \
    ++aug_mode=no_shift \
    ++output_dir=outputs/ag_yeast_sweep/flatten_512_512_hlr \
    ++cache_dir=outputs/ag_yeast_sweep/embedding_cache \
    ++seed=42 \
    ++wandb_mode=online
