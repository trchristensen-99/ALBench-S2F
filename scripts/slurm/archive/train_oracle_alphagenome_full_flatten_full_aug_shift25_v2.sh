#!/bin/bash
# boda-flatten-512-512 + full aug with max_shift=25 and detach_backbone=True fix.
# Tests whether a wider shift window (±25 vs ±15 bp) improves over fixed detach run.
#SBATCH --job-name=ag_flatten_aug_s25v2
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=kooq
#SBATCH --qos=koolab
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=200G

source /etc/profile.d/modules.sh
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD:$PYTHONPATH"
source scripts/slurm/setup_hpc_deps.sh

export XLA_FLAGS="${XLA_FLAGS} --xla_gpu_enable_command_buffer="

uv run python experiments/train_oracle_alphagenome_full.py \
    ++head_arch="boda-flatten-512-512" \
    ++aug_mode="full" \
    ++gpu=0 \
    ++seed=42 \
    ++output_dir=outputs/ag_flatten_full_aug_shift25_v2 \
    ++dropout_rate=0.1 \
    ++lr_schedule=none \
    ++max_shift=25 \
    ++epochs=100
