#!/bin/bash
# Comprehensive batch size × learning rate grid for cached head training.
# Tests BS ∈ {32, 128, 512, 1024, 4096} × LR ∈ {3e-4, 1e-3, 3e-3, 8e-3, 1.5e-2}
# = 25 configs. Measures speed (s/epoch) and full test performance.
#
# Note: test eval takes ~49 min per task (AlphaGenome JIT compilation + 71K inference).
# Training itself is 1-15 min depending on batch size.
#
# Submit:
#   /cm/shared/apps/slurm/current/bin/sbatch scripts/slurm/test_bs_lr_grid.sh
#
#SBATCH --job-name=bs_lr_grid
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=200G
#SBATCH --array=0-24

set -euo pipefail

set +u
source /etc/profile.d/modules.sh
set -u
module load EB5

cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
source scripts/slurm/setup_hpc_deps.sh

export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_command_buffer= --xla_gpu_autotune_level=0"

# Grid: 5 batch sizes × 5 learning rates = 25 configs
BATCH_SIZES=(32 128 512 1024 4096)
LRS=(0.0003 0.001 0.003 0.008 0.015)
N_LRS=${#LRS[@]}

BS_IDX=$((SLURM_ARRAY_TASK_ID / N_LRS))
LR_IDX=$((SLURM_ARRAY_TASK_ID % N_LRS))
BS=${BATCH_SIZES[$BS_IDX]}
LR=${LRS[$LR_IDX]}
TAG="bs${BS}_lr${LR}"

OUT_BASE="outputs/bs_lr_grid"

echo "=== BS=${BS} LR=${LR} (${TAG}) task=${SLURM_ARRAY_TASK_ID} ==="
echo "Node: ${SLURMD_NODENAME}  Date: $(date)"

uv run --no-sync python experiments/train_oracle_alphagenome_yeast.py \
  --config-name oracle_alphagenome_yeast_finetune_sweep \
  "++wandb_mode=offline" \
  "++batch_size=${BS}" \
  "++lr=${LR}" \
  "++epochs=20" \
  "++early_stop_patience=20" \
  "++second_stage_lr=null" \
  "++seed=42" \
  "++output_dir=${OUT_BASE}/${TAG}"

echo "=== ${TAG} DONE at $(date) ==="
