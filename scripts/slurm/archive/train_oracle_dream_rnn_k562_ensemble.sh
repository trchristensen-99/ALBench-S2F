#!/bin/bash
# Train K562 DREAM-RNN oracle ensemble (10 folds).
# Each array task trains one fold with a different seed.
# Checkpoints saved to outputs/oracle_dream_rnn_k562_ensemble/oracle_N/best_model.pt
#
#SBATCH --job-name=oracle_dream_k562
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=64G
#SBATCH --array=0-9

set -euo pipefail

set +u
source /etc/profile.d/modules.sh
set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
source scripts/slurm/setup_hpc_deps.sh

FOLD=$SLURM_ARRAY_TASK_ID
SEED=$((42 + FOLD * 7919))
OUTPUT_DIR="outputs/oracle_dream_rnn_k562_ensemble/oracle_${FOLD}"

echo "=== K562 DREAM-RNN Oracle: fold=${FOLD}, seed=${SEED} ==="
echo "Node: $SLURMD_NODENAME  Date: $(date)"

uv run --no-sync python experiments/train_oracle_dream_rnn_k562.py \
    --config-name oracle_dream_rnn_k562 \
    ++output_dir="${OUTPUT_DIR}" \
    ++seed="${SEED}" \
    ++wandb_mode=offline

echo "Done: $(date)"
