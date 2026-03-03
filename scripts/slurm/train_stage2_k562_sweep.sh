#!/bin/bash
# Stage 2 encoder fine-tuning sweep on K562 hashFrag (3 LR configs).
#
# Each task runs one Stage 2 config:
#   0 → s2a: encoder_lr=1e-5, head_lr=1e-5  (reference uniform)
#   1 → s2b: encoder_lr=1e-5, head_lr=1e-3  (differential)
#   2 → s2c: encoder_lr=1e-4, head_lr=1e-3  (aggressive)
#
# Prerequisites:
#   Stage 1 oracle_0 must be trained:
#     outputs/ag_hashfrag_oracle_cached/oracle_0/best_model/checkpoint
#
# Submit:
#   sbatch scripts/slurm/train_stage2_k562_sweep.sh
#
#SBATCH --job-name=ag_stage2_sweep
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=200G
#SBATCH --array=0-2

CONFIGS=(stage2_k562_s2a stage2_k562_s2b stage2_k562_s2c)
CONFIG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"

source /etc/profile.d/modules.sh
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD:$PYTHONPATH"
source scripts/slurm/setup_hpc_deps.sh

export XLA_FLAGS="${XLA_FLAGS} --xla_gpu_enable_command_buffer= --xla_gpu_autotune_level=0"

echo "Starting Stage 2 sweep (task ${SLURM_ARRAY_TASK_ID}/2, config=${CONFIG}) on $(date)"
echo "Node: ${SLURMD_NODENAME}"

uv run --no-sync python experiments/train_stage2_k562_hashfrag.py \
    --config-name "${CONFIG}" \
    ++wandb_mode=offline
