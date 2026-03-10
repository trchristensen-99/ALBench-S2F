#!/bin/bash
# Batch size validation for yeast AG cached head-only training.
# Tests 128, 256, 512, 1024 for 20 epochs each to confirm large batches are safe.
#
# Submit:
#   /cm/shared/apps/slurm/current/bin/sbatch scripts/slurm/test_batch_size_yeast_ag.sh
#
#SBATCH --job-name=ag_yeast_bsz_test
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=fast
#SBATCH --time=04:00:00
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

# All tasks use MPRA baseline config, only batch_size varies.
COMMON_ARGS=(
  "--config-name" "oracle_alphagenome_yeast_finetune_sweep"
  "++cache_dir=outputs/ag_yeast/embedding_cache"
  "++wandb_mode=offline"
  "++second_stage_lr=null"
  "++epochs=20"
  "++early_stop_patience=100"
  "++seed=42"
)

OUT_BASE="outputs/ag_yeast_bsz_test"
BATCH_SIZES=(128 256 512 1024)

BSZ="${BATCH_SIZES[${SLURM_ARRAY_TASK_ID}]}"

echo "Starting batch size test: BSZ=${BSZ}, task=${SLURM_ARRAY_TASK_ID}"
echo "Node: ${SLURMD_NODENAME}  Date: $(date)"

uv run --no-sync python experiments/train_oracle_alphagenome_yeast.py \
  "${COMMON_ARGS[@]}" \
  "++batch_size=${BSZ}" \
  "++output_dir=${OUT_BASE}/bsz_${BSZ}"
