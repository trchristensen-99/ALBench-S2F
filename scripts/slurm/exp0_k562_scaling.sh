#!/bin/bash
#SBATCH --job-name=exp0_k562
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=36:00:00
#SBATCH --array=0-6
#SBATCH --output=logs/exp0_k562_%a_%j.out
#SBATCH --error=logs/exp0_k562_%a_%j.err

source /etc/profile.d/modules.sh
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
source setup_env.sh

FRACTIONS=(0.01 0.02 0.05 0.10 0.20 0.50 1.00)
FRACTION=${FRACTIONS[$SLURM_ARRAY_TASK_ID]}

echo "Job ID: $SLURM_JOB_ID task=$SLURM_ARRAY_TASK_ID fraction=$FRACTION"
echo "Node: $SLURMD_NODENAME"
echo "Date: $(date)"

~/.local/bin/uv sync --extra dev
~/.local/bin/uv run --no-sync python experiments/exp0_k562_scaling.py \
  fraction="$FRACTION" \
  data_path=data/k562 \
  output_dir=outputs/exp0_k562_scaling \
  wandb_mode="${WANDB_MODE:-offline}"
