#!/bin/bash
#SBATCH --job-name=oracle_dream_rnn_k562
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=logs/oracle_dream_rnn_k562_%j.out
#SBATCH --error=logs/oracle_dream_rnn_k562_%j.err

source /etc/profile.d/modules.sh
module load EB5
source setup_env.sh

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Date: $(date)"

~/.local/bin/uv sync --extra dev
~/.local/bin/uv run --no-sync python experiments/train_oracle_dream_rnn_k562.py \
  data_path=data/k562 \
  output_dir=outputs/oracle_dream_rnn_k562 \
  epochs=80 \
  wandb_mode="${WANDB_MODE:-offline}"
