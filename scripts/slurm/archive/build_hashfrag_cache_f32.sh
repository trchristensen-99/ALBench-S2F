#!/bin/bash
# Build float32 AlphaGenome embedding cache for K562 hashFrag splits (train + val).
# Fixes float16 truncation that zeroed ~60% of embeddings (bfloat16 → float16 precision loss).
#
# Submit: /cm/shared/apps/slurm/current/bin/sbatch scripts/slurm/build_hashfrag_cache_f32.sh
#
#SBATCH --job-name=ag_hf_cache_f32
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --time=12:00:00
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

echo "Building float32 hashFrag embedding cache on ${SLURMD_NODENAME} ($(date))"
echo "Output: outputs/ag_hashfrag/embedding_cache_f32/"

uv run --no-sync python scripts/analysis/build_hashfrag_embedding_cache.py \
    --data_path data/k562 \
    --cache_dir outputs/ag_hashfrag/embedding_cache_f32 \
    --splits train val \
    --dtype float32 \
    --batch_size 128 \
    --num_workers 8

echo "Done — $(date)"
