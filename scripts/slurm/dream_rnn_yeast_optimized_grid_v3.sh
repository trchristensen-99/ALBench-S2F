#!/bin/bash
# DREAM-RNN yeast grid v3: small batch sizes (32, 64).
#
# v1 (bs=128,512 × lr=0.003,0.005 × do=0.3,0.5) is outperforming v2.
# This grid tests whether even smaller batch sizes help.
#
# Grid: 2 bs × 2 lr × 2 dropout = 8 configs, seed=42, f=1.0
#
# Submit:
#   /cm/shared/apps/slurm/current/bin/sbatch scripts/slurm/dream_rnn_yeast_optimized_grid_v3.sh
#
#SBATCH --job-name=dream_yeast_v3
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-7

set -euo pipefail

set +u
source /etc/profile.d/modules.sh
set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
source scripts/slurm/setup_hpc_deps.sh

# Grid: 2 bs × 2 lr × 2 dropout_lstm = 8
BATCH_SIZES=(32 64)
LRS=(0.003 0.005)
DROPOUTS=(0.3 0.5)

N_BS=${#BATCH_SIZES[@]}
N_LR=${#LRS[@]}

BS_IDX=$((SLURM_ARRAY_TASK_ID % N_BS))
LR_IDX=$(((SLURM_ARRAY_TASK_ID / N_BS) % N_LR))
DO_IDX=$((SLURM_ARRAY_TASK_ID / (N_BS * N_LR)))

BS=${BATCH_SIZES[$BS_IDX]}
LR=${LRS[$LR_IDX]}
DO=${DROPOUTS[$DO_IDX]}
TAG="bs${BS}_lr${LR}_do${DO}"

OUT_DIR="outputs/dream_yeast_optimized_grid_v3/${TAG}"

echo "=== DREAM-RNN yeast optimized v3: ${TAG} (task ${SLURM_ARRAY_TASK_ID}) ==="
echo "Node: ${SLURMD_NODENAME}  Date: $(date)"

# Skip if result already exists
if [ -f "${OUT_DIR}/seed_42/scaling_curve.json" ]; then
    echo "SKIP: result already exists"
    exit 0
fi

uv run --no-sync python experiments/exp0_yeast_scaling.py \
    fraction=1.0 \
    output_dir="${OUT_DIR}" \
    batch_size="${BS}" \
    lr="${LR}" \
    lr_lstm="${LR}" \
    dropout_lstm="${DO}" \
    dropout_cnn=0.2 \
    epochs=30 \
    early_stopping_patience=10 \
    seed=42 \
    num_workers=4 \
    test_subset_dir=data/yeast/test_subset_ids \
    wandb_mode=offline

echo "=== ${TAG} DONE — $(date) ==="
