#!/bin/bash
#SBATCH --job-name=ag_yeast_f512x512x256
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

# 3-layer, 512 → 512 → 256 (wide then gradual compression)
uv run python experiments/train_oracle_alphagenome_yeast.py \
    ++head_arch="flatten-mlp" \
    "++hidden_dims=[512,512,256]" \
    ++dropout_rate=0.1 \
    ++lr=1e-4 \
    ++weight_decay=1e-4 \
    ++lr_schedule=plateau \
    ++aug_mode=no_shift \
    ++output_dir=outputs/ag_yeast_sweep/flatten_512x512x256 \
    ++cache_dir=outputs/ag_yeast_sweep/embedding_cache \
    ++seed=42 \
    ++wandb_mode=online
