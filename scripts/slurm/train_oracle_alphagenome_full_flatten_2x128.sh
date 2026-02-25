#!/bin/bash
# flatten-mlp: 2 hidden layers, 128x128 units. Architecture search (depth × width grid).
#SBATCH --job-name=ag_flatten_2x128
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=kooq
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --qos=koolab
#SBATCH --cpus-per-task=14
#SBATCH --mem=200G

source /etc/profile.d/modules.sh
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD:$PYTHONPATH"
source scripts/slurm/setup_hpc_deps.sh

export XLA_FLAGS="${XLA_FLAGS} --xla_gpu_enable_command_buffer="

uv run python experiments/train_oracle_alphagenome_full.py \
    ++head_arch="flatten-mlp" \
    ++hidden_dims=[128,128] \
    ++aug_mode="no_shift" \
    ++gpu=0 \
    ++seed=42 \
    ++output_dir=outputs/ag_flatten_2x128 \
    ++cache_dir=outputs/ag_flatten/embedding_cache \
    ++dropout_rate=0.1 \
    ++lr_schedule=none \
    ++epochs=50
