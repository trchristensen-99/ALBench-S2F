#!/bin/bash
# DREAM-RNN yeast hyperparameter grid search.
# Tests batch_size × dropout_lstm × lr on f=0.1 (600K samples) for speed.
# Grid: 3 × 3 × 3 = 27 configs, each run once.
#
# Key hypotheses:
#   - batch_size=128 was much better than 1024 on K562 (0.82 → 0.88)
#   - dropout_lstm=0.5 may be too high (leaving performance on the table)
#   - lr interacts with batch_size (smaller bs may prefer smaller lr)
#
# Submit:
#   /cm/shared/apps/slurm/current/bin/sbatch scripts/slurm/dream_rnn_yeast_grid_search.sh
#
#SBATCH --job-name=dream_yeast_grid
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-26

set -euo pipefail

set +u
source /etc/profile.d/modules.sh
set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
source scripts/slurm/setup_hpc_deps.sh

# Grid dimensions
BATCH_SIZES=(128 512 1024)
DROPOUT_LSTMS=(0.2 0.3 0.5)
LRS=(0.001 0.003 0.005)

# Map array index to (bs, dropout, lr)
N_BS=${#BATCH_SIZES[@]}
N_DO=${#DROPOUT_LSTMS[@]}
N_LR=${#LRS[@]}

BS_IDX=$((SLURM_ARRAY_TASK_ID % N_BS))
DO_IDX=$(((SLURM_ARRAY_TASK_ID / N_BS) % N_DO))
LR_IDX=$((SLURM_ARRAY_TASK_ID / (N_BS * N_DO)))

BS=${BATCH_SIZES[$BS_IDX]}
DO=${DROPOUT_LSTMS[$DO_IDX]}
LR=${LRS[$LR_IDX]}

echo "Yeast grid search: bs=${BS} dropout_lstm=${DO} lr=${LR} (task ${SLURM_ARRAY_TASK_ID})"
echo "Node: $SLURMD_NODENAME  Date: $(date)"

uv run --no-sync python experiments/exp0_yeast_scaling.py \
    fraction=0.1 \
    output_dir="outputs/dream_yeast_grid/bs${BS}_do${DO}_lr${LR}" \
    batch_size="${BS}" \
    dropout_lstm="${DO}" \
    lr="${LR}" \
    lr_lstm="${LR}" \
    num_workers=4 \
    test_subset_dir=data/yeast/test_subset_ids \
    seed=42 \
    wandb_mode=offline

echo "bs=${BS} do=${DO} lr=${LR} DONE — $(date)"
