#!/bin/bash
# Generate AlphaGenome oracle ensemble pseudo-labels using float32 caches.
#
# Prerequisites:
#   1. f32 train/val cache: outputs/ag_hashfrag/embedding_cache_f32/
#   2. f32 test caches: outputs/ag_hashfrag/embedding_cache_f32/test_*.npy
#   3. 10 oracle folds: outputs/ag_hashfrag_oracle_cached_f32/oracle_{0-9}/
#
# Submit (with dependency on oracle training):
#   /cm/shared/apps/slurm/current/bin/sbatch --dependency=afterok:$ORACLE_JOB \
#       scripts/slurm/generate_pseudolabels_k562_f32.sh
#
#SBATCH --job-name=pseudolabels_k562_f32
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G

set -euo pipefail

set +u
source /etc/profile.d/modules.sh
set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
source scripts/slurm/setup_hpc_deps.sh

export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_command_buffer= --xla_gpu_autotune_level=0"

echo "Generating K562 oracle pseudolabels (f32 cache) on ${SLURMD_NODENAME} ($(date))"

uv run --no-sync python experiments/generate_oracle_pseudolabels_k562_ag.py \
    ++oracle_dir=outputs/ag_hashfrag_oracle_cached_f32 \
    ++cache_dir=outputs/ag_hashfrag/embedding_cache_f32 \
    ++output_dir=outputs/oracle_pseudolabels_k562_ag_f32 \
    ++wandb_mode=disabled

echo "Done — $(date)"
