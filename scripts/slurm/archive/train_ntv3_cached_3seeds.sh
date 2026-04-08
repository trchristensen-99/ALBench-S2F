#!/bin/bash
# Train NTv3 650M head on cached embeddings. 3 random seeds.
# Depends on: build_ntv3_embedding_cache.sh completing first.
#
# Submit:
#   /cm/shared/apps/slurm/current/bin/sbatch scripts/slurm/train_ntv3_cached_3seeds.sh
#
#SBATCH --job-name=ntv3_head
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-2

set -euo pipefail

set +u
source /etc/profile.d/modules.sh
set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
source scripts/slurm/setup_hpc_deps.sh

echo "NTv3 head training: seed_idx=${SLURM_ARRAY_TASK_ID}"
echo "Node: ${SLURMD_NODENAME}  Date: $(date)"

uv run --no-sync python experiments/train_foundation_cached.py \
    ++model_name=ntv3 \
    ++cache_dir=outputs/ntv3_k562_cached/embedding_cache \
    ++embed_dim=1536 \
    ++output_dir=outputs/ntv3_k562_cached

echo "seed_idx=${SLURM_ARRAY_TASK_ID} DONE — $(date)"
