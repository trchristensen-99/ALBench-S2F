#!/bin/bash
# flatten-mlp: 1 hidden layer, 512 units. Architecture search (depth × width grid).
#SBATCH --job-name=ag_flatten_1x512
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpuq
#SBATCH --time=04:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1

source /etc/profile.d/modules.sh
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD:$PYTHONPATH"
source scripts/slurm/setup_hpc_deps.sh

export XLA_FLAGS="${XLA_FLAGS} --xla_gpu_enable_command_buffer="

uv run python experiments/train_oracle_alphagenome_full.py \
    ++head_arch="flatten-mlp" \
    ++hidden_dims=[512] \
    ++aug_mode="no_shift" \
    ++gpu=0 \
    ++seed=42 \
    ++output_dir=outputs/ag_flatten_1x512 \
    ++cache_dir=outputs/ag_flatten/embedding_cache \
    ++dropout_rate=0.1 \
    ++lr_schedule=none \
    ++epochs=50
