#!/bin/bash
# Build AlphaGenome encoder embedding cache for the K562 test set (chr 7, 13).
# Run once; subsequent eval_ag_chrom_test.py calls with --cache_dir skip the encoder (~10x faster).
#SBATCH --job-name=ag_test_cache
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpuq
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1

set -e
source /etc/profile.d/modules.sh
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD:$PYTHONPATH"
source scripts/slurm/setup_hpc_deps.sh

uv run python scripts/analysis/build_test_embedding_cache.py \
    --data_path data/k562 \
    --cache_dir outputs/ag_flatten/embedding_cache \
    --seq_len 600 \
    --batch_size 128 \
    --num_workers 4
