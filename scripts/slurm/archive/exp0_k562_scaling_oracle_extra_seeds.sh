#!/bin/bash
# Exp 0: Extra DREAM-RNN oracle-label replicates for small K562 fractions.
# 5 seeds × 5 fractions (0.01, 0.02, 0.05, 0.10, 0.20) = 25 tasks.
# Seeds are random (seed: null in config).
#
# Submit:
#   /cm/shared/apps/slurm/current/bin/sbatch scripts/slurm/exp0_k562_scaling_oracle_extra_seeds.sh
#
#SBATCH --job-name=exp0_oracle_extra
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=36:00:00
#SBATCH --array=0-24

set -euo pipefail

set +u
source /etc/profile.d/modules.sh
set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
source scripts/slurm/setup_hpc_deps.sh

FRACTIONS=(0.01 0.02 0.05 0.10 0.20)
N_FRACTIONS=${#FRACTIONS[@]}

SEED_IDX=$(( SLURM_ARRAY_TASK_ID / N_FRACTIONS ))
FRAC_IDX=$(( SLURM_ARRAY_TASK_ID % N_FRACTIONS ))
FRACTION=${FRACTIONS[$FRAC_IDX]}

echo "DREAM-RNN oracle-label extra seeds: fraction=${FRACTION} seed_idx=${SEED_IDX} (task ${SLURM_ARRAY_TASK_ID})"
echo "Node: ${SLURMD_NODENAME}  Date: $(date)"

uv run --no-sync python experiments/exp0_k562_scaling_oracle_labels.py \
    ++fraction="$FRACTION" \
    ++wandb_mode=offline

echo "fraction=${FRACTION} seed_idx=${SEED_IDX} DONE — $(date)"
