#!/bin/bash
# Train 10-fold AlphaGenome oracle on K562 using float32 embedding cache.
# Each fold writes to outputs/ag_hashfrag_oracle_cached_f32/oracle_{fold}/.
#
# Prerequisites:
#   Build f32 cache: sbatch scripts/slurm/build_hashfrag_cache_f32.sh
#
# Submit (with dependency):
#   /cm/shared/apps/slurm/current/bin/sbatch --dependency=afterok:$CACHE_JOB \
#       scripts/slurm/train_oracle_hashfrag_f32_array.sh
#
#SBATCH --job-name=ag_oracle_f32
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --array=0-9

set -euo pipefail

set +u
source /etc/profile.d/modules.sh
set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
source scripts/slurm/setup_hpc_deps.sh

export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_command_buffer= --xla_gpu_autotune_level=0"

echo "Starting f32 cached oracle fold ${SLURM_ARRAY_TASK_ID}/9 on ${SLURMD_NODENAME} ($(date))"

uv run --no-sync python experiments/train_oracle_alphagenome_hashfrag_cached.py \
    ++fold_id=${SLURM_ARRAY_TASK_ID} \
    ++cache_dir=outputs/ag_hashfrag/embedding_cache_f32 \
    ++output_dir="outputs/ag_hashfrag_oracle_cached_f32/oracle_${SLURM_ARRAY_TASK_ID}" \
    ++wandb_mode=offline

echo "Fold ${SLURM_ARRAY_TASK_ID} done — $(date)"
