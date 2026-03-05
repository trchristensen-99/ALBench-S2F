#!/bin/bash
# Batch size test for yeast AG Stage 1 (cached head-only) training.
# Tests batch_size = 512, 1024, 4096, 16384, 65536, 200000 on full 200K cache.
# Trains for 20 epochs (with early stopping patience=20 to ensure all run).
# Measures: time/epoch, val Pearson R, train loss.
#
# Submit:
#   /cm/shared/apps/slurm/current/bin/sbatch scripts/slurm/test_batch_size_yeast_s1_cached.sh
#
#SBATCH --job-name=bs_test_s1
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=fast
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=200G
#SBATCH --array=0-5

set -euo pipefail

set +u
source /etc/profile.d/modules.sh
set -u
module load EB5

cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
source scripts/slurm/setup_hpc_deps.sh

export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_command_buffer= --xla_gpu_autotune_level=0"

BATCH_SIZES=(512 1024 4096 16384 65536 200000)
BS=${BATCH_SIZES[$SLURM_ARRAY_TASK_ID]}

OUT_BASE="outputs/batch_size_test_yeast_s1"

echo "=== S1 Cached Batch Size Test: BS=${BS} task=${SLURM_ARRAY_TASK_ID} ==="
echo "Node: ${SLURMD_NODENAME}  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)  Date: $(date)"

uv run --no-sync python experiments/train_oracle_alphagenome_yeast.py \
  --config-name oracle_alphagenome_yeast_finetune_sweep \
  "++wandb_mode=offline" \
  "++batch_size=${BS}" \
  "++epochs=20" \
  "++early_stop_patience=20" \
  "++second_stage_lr=null" \
  "++seed=42" \
  "++output_dir=${OUT_BASE}/bs_${BS}"

echo "=== S1 Cached Batch Size Test BS=${BS} DONE at $(date) ==="
