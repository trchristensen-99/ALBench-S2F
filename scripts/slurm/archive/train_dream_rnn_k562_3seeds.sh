#!/bin/bash
# Train 3 DREAM-RNN models on full K562 HashFrag train set.
# Uses batch_size=128, lr=0.005 (best from hyperparameter sweep).
# 3 random seeds, each evaluated on all 4 test metrics.
#
# Submit:
#   /cm/shared/apps/slurm/current/bin/sbatch scripts/slurm/train_dream_rnn_k562_3seeds.sh
#
#SBATCH --job-name=dream_k562_3s
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=36:00:00
#SBATCH --array=0-2

set -euo pipefail

set +u
source /etc/profile.d/modules.sh
set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PATH=$PWD/external/hashFrag/src:$PWD/external/hashFrag:$PATH
source scripts/slurm/setup_hpc_deps.sh

echo "DREAM-RNN K562 training (bs=128, lr=0.005): seed_idx=${SLURM_ARRAY_TASK_ID}"
echo "Node: ${SLURMD_NODENAME}  Date: $(date)"

uv run --no-sync python experiments/exp0_k562_scaling.py \
    fraction=1.0 \
    data_path=data/k562 \
    output_dir=outputs/dream_rnn_k562_3seeds \
    batch_size=128 \
    lr=0.005 \
    lr_lstm=0.005 \
    wandb_mode=offline

echo "seed_idx=${SLURM_ARRAY_TASK_ID} DONE — $(date)"
