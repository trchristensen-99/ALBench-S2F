#!/bin/bash
# Build FULL AlphaGenome embedding cache for yeast (all 6M+ train sequences).
# Replaces the 200K-limited cache used previously.
# Outputs ~126GB to outputs/ag_yeast/embedding_cache_full/.
#
# Optimized: BS=512 + concatenated canonical/RC forward passes (effective GPU BS=1024).
# Estimated runtime: ~15-25h on H100.
#
# After this completes, submit AG yeast scaling:
#   /cm/shared/apps/slurm/current/bin/sbatch --dependency=afterok:$JOBID scripts/slurm/exp0_yeast_scaling_alphagenome_full.sh
#
# Submit:
#   /cm/shared/apps/slurm/current/bin/sbatch scripts/slurm/build_yeast_embedding_cache_full.sh
#
#SBATCH --job-name=ag_yeast_cache_full
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --time=48:00:00
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

echo "Building FULL yeast embedding cache on ${SLURMD_NODENAME} at $(date)"
echo "Splits: train val test | Cache: outputs/ag_yeast/embedding_cache_full/"
echo "Disk free before:"
df -h /grid/wsbs/home_norepl/christen | tail -1

uv run --no-sync python scripts/analysis/build_yeast_embedding_cache.py \
    --data_path data/yeast \
    --cache_dir outputs/ag_yeast/embedding_cache_full \
    --splits train val test \
    --batch_size 512 \
    --num_workers 12

echo "Disk free after:"
df -h /grid/wsbs/home_norepl/christen | tail -1
echo "Done at $(date)"
