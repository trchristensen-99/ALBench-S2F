#!/bin/bash
#SBATCH --job-name=ag_yeast_flat_2s
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpuq
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --constraint=h100
#SBATCH --time=12:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=christen@cshl.edu

source /etc/profile.d/modules.sh
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD:$PYTHONPATH"
mkdir -p logs

# boda-flatten-512-512, lr=1e-3 (Stage 1 head-only) + Stage 2 unfreeze at lr=1e-5.
# Stage 1: no_shift cached mode, 50 epoch budget, early_stop=10.
# Stage 2: full augmentation (encoder runs every step), 30 epochs, early_stop=10.
uv run python experiments/train_oracle_alphagenome_yeast.py \
    ++head_arch="boda-flatten-512-512" \
    ++dropout_rate=0.1 \
    ++lr=0.001 \
    ++weight_decay=1e-6 \
    ++lr_schedule=none \
    ++aug_mode=no_shift \
    ++epochs=50 \
    ++early_stop_patience=10 \
    ++second_stage_lr=1e-5 \
    ++second_stage_epochs=30 \
    ++second_stage_early_stop_patience=10 \
    ++output_dir=outputs/ag_yeast_sweep/flatten_512_512_2stage \
    ++cache_dir=outputs/ag_yeast_sweep/embedding_cache \
    ++seed=42 \
    ++wandb_mode=online \
    ++batch_size=4096
