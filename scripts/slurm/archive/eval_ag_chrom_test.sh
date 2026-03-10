#!/bin/bash
# Evaluate AlphaGenome Boda heads (sum, mean, max, center) on chr 7, 13 test set.
# Writes outputs/ag_chrom_test_results.json for comparison with Malinois.
#
# Builds float32 test embedding cache on first run (idempotent; ~5–10 min on H100).
# Full-aug heads use _F32_CACHE to avoid float16 clipping of bfloat16 encoder output.
#SBATCH --job-name=ag_chrom_test
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpuq
#SBATCH --time=04:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1

set -e
source /etc/profile.d/modules.sh
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD:$PYTHONPATH"
source scripts/slurm/setup_hpc_deps.sh

# Disable XLA CUDA command buffers to prevent OOM when accumulating too many live CUDA graphs.
export XLA_FLAGS="${XLA_FLAGS} --xla_gpu_enable_command_buffer="

# Build float32 test cache (skipped automatically if already exists).
# Required for full_aug models trained on unclipped bfloat16 encoder output.
uv run python scripts/analysis/build_test_embedding_cache.py \
    --dtype float32 \
    --data_path data/k562 \
    --cache_dir outputs/ag_flatten/embedding_cache_f32

uv run python scripts/analysis/eval_ag_chrom_test.py \
    --data_path data/k562 \
    --output outputs/ag_chrom_test_results.json \
    --cache_dir outputs/ag_flatten/embedding_cache
