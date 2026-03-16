#!/bin/bash
# Build Enformer embedding cache for K562 train split using SLURM array sharding.
# 4 GPUs process 1/4 of train sequences each in parallel, saving shard files.
# Val + test sets are NOT built here (handled by merge_and_sweep job).
#
# Submit:
#   /cm/shared/apps/slurm/current/bin/sbatch scripts/slurm/build_enformer_cache_sharded.sh
#
#SBATCH --job-name=enformer_cache
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=16:00:00
#SBATCH --array=0-3

set -euo pipefail

set +u
source /etc/profile.d/modules.sh
set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
source scripts/slurm/setup_hpc_deps.sh

N_SHARDS=4
SHARD_IDX=${SLURM_ARRAY_TASK_ID}

echo "Building Enformer cache shard ${SHARD_IDX}/${N_SHARDS} — $(date)"
echo "Node: ${SLURMD_NODENAME}"

uv run --no-sync python scripts/build_enformer_embedding_cache.py \
    --data-path data/k562 \
    --cache-dir outputs/enformer_k562_cached/embedding_cache \
    --splits train \
    --batch-size 4 \
    --shard-idx "${SHARD_IDX}" \
    --n-shards "${N_SHARDS}"

echo "Enformer cache shard ${SHARD_IDX} DONE — $(date)"
