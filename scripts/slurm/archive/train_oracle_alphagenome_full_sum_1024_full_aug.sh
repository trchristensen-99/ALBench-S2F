#!/bin/bash
# boda-sum-1024-dropout head + full shift augmentation.
# Tests whether the larger 1024-unit sum head benefits from shift aug.
# Compares against: ag_sum_full_aug (512-512) and ag_sum_1024_ref (no_shift).
#SBATCH --job-name=ag_sum_1024_full_aug
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
    ++head_arch="boda-sum-1024-dropout" \
    ++aug_mode="full" \
    ++gpu=0 \
    ++seed=42 \
    ++output_dir=outputs/ag_sum_1024_full_aug \
    ++dropout_rate=0.1 \
    ++lr_schedule=none \
    ++epochs=100
