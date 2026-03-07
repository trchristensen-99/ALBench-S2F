#!/bin/bash
# Build Enformer embedding cache for K562 hashFrag data.
# Enformer is slow (~12-24h on H100, batch_size=4). Includes train, val, and test sets.
#
# Submit:
#   /cm/shared/apps/slurm/current/bin/sbatch scripts/slurm/build_enformer_embedding_cache.sh
#
#SBATCH --job-name=enformer_cache
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00

set -euo pipefail

set +u
source /etc/profile.d/modules.sh
set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
source scripts/slurm/setup_hpc_deps.sh

echo "Building Enformer embedding cache — $(date)"
echo "Node: ${SLURMD_NODENAME}"

uv run --no-sync python scripts/build_enformer_embedding_cache.py \
    --data-path data/k562 \
    --cache-dir outputs/enformer_k562_cached/embedding_cache \
    --splits train val \
    --include-test \
    --batch-size 4

echo "Enformer cache DONE — $(date)"
