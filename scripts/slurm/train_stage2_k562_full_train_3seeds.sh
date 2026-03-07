#!/bin/bash
# Train 3 AlphaGenome Stage 2 models on the full K562 hashFrag train split.
# Uses s2c hyperparameters (encoder_lr=1e-4, head_lr=1e-3).
# Starts from S1 head trained on full train split (f=1.0).
# Validates on dedicated val split (not k-fold CV).
#
# Submit:
#   /cm/shared/apps/slurm/current/bin/sbatch scripts/slurm/train_stage2_k562_full_train_3seeds.sh
#
#SBATCH --job-name=ag_s2_full
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=200G
#SBATCH --time=48:00:00
#SBATCH --array=0-2

set -euo pipefail

set +u
source /etc/profile.d/modules.sh
set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
source scripts/slurm/setup_hpc_deps.sh

echo "AG Stage 2 full-train: seed_idx=${SLURM_ARRAY_TASK_ID}"
echo "Node: ${SLURMD_NODENAME}  Date: $(date)"

uv run --no-sync python experiments/train_stage2_k562_hashfrag.py \
    --config-name stage2_k562_full_train \
    ++output_dir="outputs/stage2_k562_full_train/run_${SLURM_ARRAY_TASK_ID}" \
    ++wandb_mode=offline

echo "seed_idx=${SLURM_ARRAY_TASK_ID} DONE — $(date)"
