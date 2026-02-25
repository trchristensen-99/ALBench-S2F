#!/bin/bash
# boda-flatten-512-512 + full aug with max_shift=25 (vs default 15).
# Tests whether wider shift augmentation window improves generalization.
#SBATCH --job-name=ag_flatten_full_aug_s25
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
    ++head_arch="boda-flatten-512-512" \
    ++aug_mode="full" \
    ++gpu=0 \
    ++seed=42 \
    ++output_dir=outputs/ag_flatten_full_aug_shift25 \
    ++dropout_rate=0.1 \
    ++lr_schedule=none \
    ++max_shift=25 \
    ++epochs=100
