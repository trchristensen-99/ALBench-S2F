#!/bin/bash
# Concatenate chunked yeast embedding cache after parallel build.
# CPU-only job — no GPU needed.
#
# Submit with dependency on chunk build job:
#   /cm/shared/apps/slurm/current/bin/sbatch --dependency=afterok:$CHUNK_JOB scripts/slurm/concat_yeast_cache.sh
#
#SBATCH --job-name=ag_yeast_cache_concat
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G

set -euo pipefail

set +u
source /etc/profile.d/modules.sh
set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
source scripts/slurm/setup_hpc_deps.sh

echo "Concatenating yeast cache chunks at $(date)"
echo "Disk free before:"
df -h /grid/wsbs/home_norepl/christen | tail -1

uv run --no-sync python scripts/analysis/concat_yeast_cache_chunks.py \
    --cache_dir outputs/ag_yeast/embedding_cache_full \
    --num_chunks 6

echo "Disk free after:"
df -h /grid/wsbs/home_norepl/christen | tail -1
echo "Done at $(date)"
