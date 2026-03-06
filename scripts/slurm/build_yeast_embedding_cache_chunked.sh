#!/bin/bash
# Build FULL AlphaGenome embedding cache for yeast in PARALLEL CHUNKS.
# Splits the 6M train sequences across NUM_CHUNKS GPU jobs.
# Each chunk builds ~1M sequences in ~12h on H100.
# After all chunks complete, a CPU job concatenates them.
#
# Submit:
#   /cm/shared/apps/slurm/current/bin/sbatch scripts/slurm/build_yeast_embedding_cache_chunked.sh
#
#SBATCH --job-name=ag_yeast_cache_chunk
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=200G
#SBATCH --array=0-5

set -euo pipefail

set +u
source /etc/profile.d/modules.sh
set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
source scripts/slurm/setup_hpc_deps.sh

export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_command_buffer= --xla_gpu_autotune_level=0"

NUM_CHUNKS=6
CACHE_DIR="outputs/ag_yeast/embedding_cache_full"

echo "Building yeast cache chunk ${SLURM_ARRAY_TASK_ID}/${NUM_CHUNKS} on ${SLURMD_NODENAME} at $(date)"
echo "Disk free before:"
df -h /grid/wsbs/home_norepl/christen | tail -1
START_TIME=$(date +%s)

# Each chunk builds its portion of train + full val/test (val/test auto-skip if already built)
uv run --no-sync python scripts/analysis/build_yeast_embedding_cache.py \
    --data_path data/yeast \
    --cache_dir "${CACHE_DIR}" \
    --splits train val test \
    --batch_size 128 \
    --num_workers 8 \
    --chunk_id "${SLURM_ARRAY_TASK_ID}" \
    --num_chunks "${NUM_CHUNKS}"

END_TIME=$(date +%s)
echo "=== chunk ${SLURM_ARRAY_TASK_ID} DONE at $(date) — wall time: $((END_TIME - START_TIME))s ==="
echo "Disk free after:"
df -h /grid/wsbs/home_norepl/christen | tail -1
