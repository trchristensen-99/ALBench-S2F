#!/bin/bash
# Build Nucleotide Transformer v2 250M embedding cache for K562 hashFrag data.
# NT is fastest (~15-30 min on H100). Includes train, val, and test sets.
#
# Submit:
#   /cm/shared/apps/slurm/current/bin/sbatch scripts/slurm/build_nt_embedding_cache.sh
#
#SBATCH --job-name=nt_cache
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00

set -euo pipefail

set +u
source /etc/profile.d/modules.sh
set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
source scripts/slurm/setup_hpc_deps.sh

echo "Building NT embedding cache — $(date)"
echo "Node: ${SLURMD_NODENAME}"

uv run --no-sync python scripts/build_nt_embedding_cache.py \
    --data-path data/k562 \
    --cache-dir outputs/nt_k562_cached/embedding_cache \
    --splits train val \
    --include-test \
    --batch-size 64

echo "NT cache DONE — $(date)"
