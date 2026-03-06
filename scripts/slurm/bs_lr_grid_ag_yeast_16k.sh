#!/bin/bash
# Test BS=16384 for AG yeast cached head training.
# Extends the BS×LR grid to verify if very large batches maintain performance.
# 3 LR configs: 1e-3, 3e-3, 1e-2. Skips test eval for speed.
#
# Submit:
#   /cm/shared/apps/slurm/current/bin/sbatch scripts/slurm/bs_lr_grid_ag_yeast_16k.sh
#
#SBATCH --job-name=bs_lr_16k
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=200G
#SBATCH --array=0-2

set -euo pipefail

set +u
source /etc/profile.d/modules.sh
set -u
module load EB5

cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
source scripts/slurm/setup_hpc_deps.sh

export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_command_buffer= --xla_gpu_autotune_level=0"

LRS=(0.001 0.003 0.01)
BS=16384
LR=${LRS[$SLURM_ARRAY_TASK_ID]}
TAG="bs${BS}_lr${LR}"

OUT_BASE="outputs/bs_lr_grid"

echo "=== BS=${BS} LR=${LR} (${TAG}) task=${SLURM_ARRAY_TASK_ID} ==="
echo "Node: ${SLURMD_NODENAME}  Date: $(date)"
START_TIME=$(date +%s)

uv run --no-sync python experiments/train_oracle_alphagenome_yeast.py \
  --config-name oracle_alphagenome_yeast_finetune_sweep \
  "++wandb_mode=offline" \
  "++batch_size=${BS}" \
  "++lr=${LR}" \
  "++epochs=20" \
  "++early_stop_patience=20" \
  "++second_stage_lr=null" \
  "++seed=42" \
  "++test_subset_dir=/nonexistent" \
  "++output_dir=${OUT_BASE}/${TAG}"

END_TIME=$(date +%s)
echo "=== ${TAG} DONE at $(date) — wall time: $((END_TIME - START_TIME))s ==="
