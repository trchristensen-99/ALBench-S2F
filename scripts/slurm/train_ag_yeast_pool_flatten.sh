#!/bin/bash
#SBATCH --job-name=ag_yeast_pool_flat
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
export WANDB_API_KEY=$(cat ~/.wandb_key)
mkdir -p logs

# pool-flatten (mean+max+flatten concat → 512→256) + DO=0.1 + plateau LR
# Previously run on yeast WITHOUT plateau or dropout; this is a proper rerun.
uv run python experiments/train_oracle_alphagenome_yeast.py \
    ++head_arch="pool-flatten" \
    ++dropout_rate=0.1 \
    ++lr_schedule=plateau \
    ++aug_mode=no_shift \
    ++output_dir=outputs/ag_yeast_sweep/pool_flatten \
    ++cache_dir=outputs/ag_yeast_sweep/embedding_cache \
    ++seed=42 \
    ++wandb_mode=online
