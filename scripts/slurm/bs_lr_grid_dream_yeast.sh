#!/bin/bash
# BS×LR grid for DREAM-RNN on yeast.
# Tests BS ∈ {128, 512, 1024, 4096, 16384} × LR ∈ {1e-3, 3e-3, 5e-3, 1e-2, 2e-2}
# = 25 configs. Uses fraction=1.0, 20 epochs, seed=42 for reproducibility.
# Default: BS=1024, lr=0.005. Test eval included (fast for DREAM-RNN).
#
# Submit:
#   /cm/shared/apps/slurm/current/bin/sbatch scripts/slurm/bs_lr_grid_dream_yeast.sh
#
#SBATCH --job-name=bs_lr_dream_yeast
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --time=2:00:00
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

# Grid: 5 batch sizes × 5 learning rates = 25 configs
BATCH_SIZES=(128 512 1024 4096 16384)
LRS=(0.001 0.003 0.005 0.01 0.02)
N_LRS=${#LRS[@]}

BS_IDX=$((SLURM_ARRAY_TASK_ID / N_LRS))
LR_IDX=$((SLURM_ARRAY_TASK_ID % N_LRS))
BS=${BATCH_SIZES[$BS_IDX]}
LR=${LRS[$LR_IDX]}
TAG="bs${BS}_lr${LR}"

echo "=== DREAM-RNN yeast BS=${BS} LR=${LR} (${TAG}) task=${SLURM_ARRAY_TASK_ID} ==="
echo "Node: ${SLURMD_NODENAME}  Date: $(date)"
START_TIME=$(date +%s)

uv run --no-sync python experiments/exp0_yeast_scaling.py \
    fraction=1.0 \
    output_dir="outputs/bs_lr_grid_dream_yeast/${TAG}" \
    batch_size="${BS}" \
    lr="${LR}" \
    lr_lstm="${LR}" \
    epochs=20 \
    seed=42 \
    num_workers=4 \
    test_subset_dir=data/yeast/test_subset_ids \
    wandb_mode=offline

END_TIME=$(date +%s)
echo "=== ${TAG} DONE at $(date) — wall time: $((END_TIME - START_TIME))s ==="
