#!/bin/bash
# Quick batch size test for yeast AG Stage 2 (full encoder) training.
# Tests batch_size = 128, 256, 512, 1024 on 20K sequences for 3 S2 epochs.
# Measures: time/iteration, time/epoch, val Pearson R, val loss.
#
# Submit:
#   /cm/shared/apps/slurm/current/bin/sbatch scripts/slurm/test_batch_size_yeast_s2.sh
#
#SBATCH --job-name=bs_test_yeast
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=fast
#SBATCH --time=4:00:00
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

BATCH_SIZES=(128 256 512 1024)
BS=${BATCH_SIZES[$SLURM_ARRAY_TASK_ID]}

OUT_BASE="outputs/batch_size_test_yeast_s2"

echo "=== Batch size test: BS=${BS} task=${SLURM_ARRAY_TASK_ID} ==="
echo "Node: ${SLURMD_NODENAME}  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)  Date: $(date)"

uv run --no-sync python experiments/train_oracle_alphagenome_yeast.py \
  --config-name oracle_alphagenome_yeast_finetune_sweep \
  "++wandb_mode=offline" \
  "++epochs=1" \
  "++early_stop_patience=100" \
  "++second_stage_lr=1e-5" \
  "++second_stage_epochs=3" \
  "++second_stage_early_stop_patience=100" \
  "++second_stage_batch_size=${BS}" \
  "++second_stage_weight_decay=1e-6" \
  "++second_stage_max_sequences=20000" \
  "++second_stage_unfreeze_mode=encoder" \
  "++output_dir=${OUT_BASE}/bs_${BS}"

echo "=== Batch size test BS=${BS} DONE at $(date) ==="
