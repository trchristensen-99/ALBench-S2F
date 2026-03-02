#!/bin/bash
# Train 10-seed AlphaGenome oracle on hashFrag K562 train split using
# precomputed encoder embedding cache (no encoder overhead per epoch).
# Each seed writes outputs/ag_hashfrag_oracle_cached/oracle_{task_id}/.
#
# Prerequisites:
#   1. Build cache: sbatch scripts/slurm/build_hashfrag_embedding_cache.sh
#   2. Submit this array: sbatch scripts/slurm/train_oracle_alphagenome_hashfrag_cached_array.sh
#
#SBATCH --job-name=ag_hf_oracle_cached
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --array=0-9

source /etc/profile.d/modules.sh
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD:$PYTHONPATH"
source scripts/slurm/setup_hpc_deps.sh

export XLA_FLAGS="${XLA_FLAGS} --xla_gpu_enable_command_buffer= --xla_gpu_autotune_level=0"

echo "Starting cached oracle (task ${SLURM_ARRAY_TASK_ID}/9) on $(date)"
echo "Node: ${SLURMD_NODENAME}"

uv run --no-sync python experiments/train_oracle_alphagenome_hashfrag_cached.py \
    ++seed=null \
    ++fold_id=${SLURM_ARRAY_TASK_ID} \
    ++output_dir="outputs/ag_hashfrag_oracle_cached/oracle_${SLURM_ARRAY_TASK_ID}" \
    ++wandb_mode=offline
