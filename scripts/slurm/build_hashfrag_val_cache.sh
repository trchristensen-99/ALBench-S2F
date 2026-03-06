#!/bin/bash
# Build ONLY the val embedding cache for K562 hashFrag.
# Quick job: val set is ~15K sequences, should take ~5-10 min on H100.
# After this completes, resubmit bs_lr_grid_ag_k562.sh.
#
# Submit:
#   /cm/shared/apps/slurm/current/bin/sbatch scripts/slurm/build_hashfrag_val_cache.sh
#
#SBATCH --job-name=build_hf_val
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=200G
#SBATCH --time=1:00:00

set -euo pipefail

set +u
source /etc/profile.d/modules.sh
set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
source scripts/slurm/setup_hpc_deps.sh

export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_command_buffer= --xla_gpu_autotune_level=0"

echo "Building hashFrag val embedding cache on $SLURMD_NODENAME ($(date))"

uv run --no-sync python scripts/analysis/build_hashfrag_embedding_cache.py \
    --data_path data/k562 \
    --cache_dir outputs/ag_hashfrag/embedding_cache \
    --splits val \
    --batch_size 128 \
    --num_workers 8

echo "Val cache built at $(date)"

# Auto-submit the AG K562 BS×LR grid after cache is built
echo "Submitting AG K562 BS×LR grid..."
/cm/shared/apps/slurm/current/bin/sbatch scripts/slurm/bs_lr_grid_ag_k562.sh
echo "AG K562 grid submitted"
