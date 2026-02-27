#!/bin/bash
# Build AlphaGenome encoder embedding cache for K562 hashFrag splits (train/pool/val).
# Run once; cache files in outputs/ag_hashfrag/embedding_cache/ are reused by
# subsequent training runs to skip the frozen encoder entirely (~20-50x speedup).
#
# Submit: sbatch scripts/slurm/build_hashfrag_embedding_cache.sh
#
#SBATCH --job-name=ag_hf_cache
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=200G
#SBATCH --time=02:00:00

source /etc/profile.d/modules.sh
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD:$PYTHONPATH"
source scripts/slurm/setup_hpc_deps.sh

# Disable XLA command buffer + cuDNN autotuner (prevents compile errors on H100)
export XLA_FLAGS="${XLA_FLAGS} --xla_gpu_enable_command_buffer= --xla_gpu_autotune_level=0"

echo "Building hashFrag embedding cache on $SLURMD_NODENAME ($(date))"
echo "Splits: train pool val  |  Output: outputs/ag_hashfrag/embedding_cache/"

uv run --no-sync python scripts/analysis/build_hashfrag_embedding_cache.py \
    --data_path data/k562 \
    --cache_dir outputs/ag_hashfrag/embedding_cache \
    --splits train pool val \
    --batch_size 128 \
    --num_workers 8

echo "Done ($(date))"
