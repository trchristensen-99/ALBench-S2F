#!/bin/bash
# Exp 0: AlphaGenome cached-head scaling curve on K562 with RC augmentation.
# Trains boda-flatten-512-512 on 7 random downsamples of hashFrag train+pool
# using pre-computed encoder embeddings with 50% RC aug per sample.
# Submit this script 3 times to get 3 independent runs per fraction:
#   for i in 1 2 3; do sbatch scripts/slurm/exp0_k562_scaling_alphagenome_cached_rcaug.sh; done
#
#SBATCH --job-name=exp0_ag_k562_cached_rcaug
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=200G
#SBATCH --array=0-6

set -euo pipefail

set +u
source /etc/profile.d/modules.sh
set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
source scripts/slurm/setup_hpc_deps.sh

export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_command_buffer= --xla_gpu_autotune_level=0"

FRACTIONS=(0.01 0.02 0.05 0.10 0.20 0.50 1.00)
FRACTION=${FRACTIONS[$SLURM_ARRAY_TASK_ID]}

echo "Starting AG cached+RC-aug scaling: fraction=${FRACTION} (task ${SLURM_ARRAY_TASK_ID}/6)"
echo "Node: $SLURMD_NODENAME  Date: $(date)"

uv run --no-sync python experiments/exp0_k562_scaling_alphagenome_cached.py \
    --config-name exp0_k562_scaling_alphagenome_cached_rcaug \
    ++fraction="${FRACTION}" \
    ++wandb_mode=offline
