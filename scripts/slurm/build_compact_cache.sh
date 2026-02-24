#!/bin/bash
# Build 384-bp compact-window encoder embeddings for K562 train + val splits.
# Submit this first; compact training jobs use --dependency=afterok:<this_job_id>.
# See scripts/slurm/submit_compact_jobs.sh for the full submission sequence.
#SBATCH --job-name=ag_compact_cache
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpuq
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1

set -e
source /etc/profile.d/modules.sh
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD:$PYTHONPATH"
source scripts/slurm/setup_hpc_deps.sh

uv run python scripts/analysis/build_compact_cache.py \
    --data_path data/k562 \
    --cache_dir outputs/ag_compact/embedding_cache_compact \
    --seq_len 384 \
    --batch_size 128 \
    --num_workers 4
