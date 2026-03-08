#!/bin/bash
# Yeast AG Stage 2 batch size sweep.
#
# Tests different S2 batch sizes at a fixed LR=1e-3 to find optimal BS.
# Best BS will be combined with best LR from v3 sweep for final config.
#
# Grid (4 tasks):
#   0 → BS=64
#   1 → BS=128
#   2 → BS=512
#   3 → BS=1024
#
# Note: BS=256 is already tested in the v3 LR sweep (task 1: lr=1e-3, enc, s1ep5).
#
# Submit:
#   /cm/shared/apps/slurm/current/bin/sbatch scripts/slurm/train_oracle_alphagenome_yeast_sweep_s2_bs.sh
#
#SBATCH --job-name=ag_yeast_s2_bs
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=200G
#SBATCH --array=0-3

set -euo pipefail

set +u
source /etc/profile.d/modules.sh
set -u
module load EB5

cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
source scripts/slurm/setup_hpc_deps.sh

export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_command_buffer= --xla_gpu_autotune_level=0"

BATCH_SIZES=(64 128 512 1024)
BS=${BATCH_SIZES[${SLURM_ARRAY_TASK_ID}]}

OUT_BASE="outputs/ag_yeast_sweep_s2_bs"

echo "=== S2 batch size sweep: task=${SLURM_ARRAY_TASK_ID} BS=${BS} ==="
echo "Node: ${SLURMD_NODENAME}  Date: $(date)"
START_TIME=$(date +%s)

uv run --no-sync python experiments/train_oracle_alphagenome_yeast.py \
  --config-name oracle_alphagenome_yeast_finetune_sweep \
  "++wandb_mode=offline" \
  "++batch_size=4096" \
  "++lr=0.003" \
  "++epochs=5" \
  "++early_stop_patience=100" \
  "++second_stage_epochs=50" \
  "++second_stage_batch_size=${BS}" \
  "++second_stage_lr=1e-3" \
  "++second_stage_weight_decay=1e-6" \
  "++second_stage_max_sequences=50000" \
  "++second_stage_early_stop_patience=7" \
  "++second_stage_unfreeze_mode=encoder" \
  "++output_dir=${OUT_BASE}/bs${BS}"

END_TIME=$(date +%s)
echo "=== task ${SLURM_ARRAY_TASK_ID} (BS=${BS}) DONE at $(date) — wall time: $((END_TIME - START_TIME))s ==="
