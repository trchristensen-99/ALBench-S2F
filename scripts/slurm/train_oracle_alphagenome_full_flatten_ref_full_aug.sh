#!/bin/bash
# Reference arch (boda-flatten-1024-dropout) + full shift augmentation.
# Combines FT_MPRA exact architecture with shift augmentation.
#SBATCH --job-name=ag_flatten_ref_full_aug
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpuq
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00

source /etc/profile.d/modules.sh
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD:$PYTHONPATH"
source scripts/slurm/setup_hpc_deps.sh

uv run python experiments/train_oracle_alphagenome_full.py \
    ++head_arch="boda-flatten-1024-dropout" \
    ++aug_mode="full" \
    ++gpu=0 \
    ++seed=42 \
    ++output_dir=outputs/ag_flatten_ref_full_aug \
    ++dropout_rate=0.1 \
    ++lr_schedule=none \
    ++epochs=100
