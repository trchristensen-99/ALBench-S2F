#!/bin/bash
# Evaluate AlphaGenome Boda heads (sum, mean, max, center) on chr 7, 13 test set.
# Writes outputs/ag_chrom_test_results.json for comparison with Malinois.
#SBATCH --job-name=ag_chrom_test
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpuq
#SBATCH --time=02:00:00
#SBATCH --mem=32G
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

uv run python scripts/analysis/eval_ag_chrom_test.py \
    --data_path data/k562 \
    --output outputs/ag_chrom_test_results.json \
    --cache_dir outputs/ag_flatten/embedding_cache
