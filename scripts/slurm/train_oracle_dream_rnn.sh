#!/bin/bash
#SBATCH --job-name=oracle_dream_rnn_yeast
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/oracle_dream_rnn_yeast_%j.out
#SBATCH --error=logs/oracle_dream_rnn_yeast_%j.err

# Load environment
source /etc/profile.d/modules.sh
module load EB5
source setup_env.sh

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Date: $(date)"

# Run training with project-managed Python (3.11+)
~/.local/bin/uv sync --extra dev
~/.local/bin/uv run --no-sync python experiments/train_oracle_dream_rnn.py \
    --data-path data/yeast \
    --output-dir outputs/oracle_dream_rnn_yeast \
    --epochs 80 \
    --wandb-mode "${WANDB_MODE:-offline}"
