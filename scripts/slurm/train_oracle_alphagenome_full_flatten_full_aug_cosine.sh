#!/bin/bash
# boda-flatten-512-512 + full shift aug + cosine LR annealing.
# Tests whether cosine LR (lr=0.001 -> 0.00001 over 50 epochs) beats
# constant LR with early stopping (668234: best val 0.9338 at epoch 17).
# early_stop_patience=50 lets the full cosine schedule play out.
#SBATCH --job-name=ag_flatten_full_aug_cosine
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00

source /etc/profile.d/modules.sh
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD:$PYTHONPATH"
source scripts/slurm/setup_hpc_deps.sh

uv run python experiments/train_oracle_alphagenome_full.py \
    ++head_arch="boda-flatten-512-512" \
    ++aug_mode="full" \
    ++gpu=0 \
    ++seed=42 \
    ++output_dir=outputs/ag_flatten_full_aug_cosine \
    ++dropout_rate=0.1 \
    ++lr_schedule=cosine \
    ++epochs=50 \
    ++early_stop_patience=50
