#!/bin/bash
# Build AlphaGenome embedding cache for yeast train/pool/val splits.
#
# Submit:
#   /cm/shared/apps/slurm/current/bin/sbatch scripts/slurm/build_yeast_embedding_cache.sh
#
#SBATCH --job-name=ag_yeast_cache
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=200G

set -euo pipefail

set +u
source /etc/profile.d/modules.sh
set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
source scripts/slurm/setup_hpc_deps.sh

export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_command_buffer= --xla_gpu_autotune_level=0"

echo "Building yeast embedding cache on ${SLURMD_NODENAME} at $(date)"
echo "Splits: train pool val | Cache: outputs/ag_yeast/embedding_cache/"

uv run --no-sync python scripts/analysis/build_yeast_embedding_cache.py \
    --data_path data/yeast \
    --cache_dir outputs/ag_yeast/embedding_cache \
    --splits train pool val \
    --batch_size 128 \
    --num_workers 8

echo "Done at $(date)"
