#!/bin/bash
# Exp 0: AlphaGenome cached-head oracle-label scaling curve on yeast.
# Trains AG head on DREAM-RNN oracle pseudolabels (not true labels).
# Uses the full 6M embedding cache.
# Multi-seed support: array index = seed_idx * N_FRACTIONS + fraction_idx.
#
# Submit 3 seeds (30 tasks):
#   /cm/shared/apps/slurm/current/bin/sbatch scripts/slurm/exp0_yeast_scaling_oracle_labels_ag.sh
#
# Submit with dependency on cache build:
#   /cm/shared/apps/slurm/current/bin/sbatch --dependency=afterok:$CACHE_JOB scripts/slurm/exp0_yeast_scaling_oracle_labels_ag.sh
#
#SBATCH --job-name=exp0_ag_yeast_oracle
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=200G
#SBATCH --array=0-29

set -euo pipefail

set +u
source /etc/profile.d/modules.sh
set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
source scripts/slurm/setup_hpc_deps.sh

export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_command_buffer= --xla_gpu_autotune_level=0"

FRACTIONS=(0.001 0.002 0.005 0.01 0.02 0.05 0.10 0.20 0.50 1.00)
N_FRACTIONS=${#FRACTIONS[@]}

# Decompose array task: seed_idx * N_FRACTIONS + fraction_idx
SEED_IDX=$(( SLURM_ARRAY_TASK_ID / N_FRACTIONS ))
FRAC_IDX=$(( SLURM_ARRAY_TASK_ID % N_FRACTIONS ))
FRACTION=${FRACTIONS[$FRAC_IDX]}

echo "Starting yeast AG oracle-label scaling: fraction=${FRACTION} seed_idx=${SEED_IDX} task=${SLURM_ARRAY_TASK_ID}"
echo "Node: ${SLURMD_NODENAME}  Date: $(date)"
START_TIME=$(date +%s)

uv run --no-sync python experiments/exp0_yeast_scaling_oracle_labels_ag.py \
    "++fraction=${FRACTION}" \
    "++wandb_mode=offline"

END_TIME=$(date +%s)
echo "=== fraction=${FRACTION} seed_idx=${SEED_IDX} DONE at $(date) — wall time: $((END_TIME - START_TIME))s ==="
