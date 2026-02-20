#!/bin/bash
#SBATCH --job-name=student_alphagenome
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/student_alphagenome_%j.out
#SBATCH --error=logs/student_alphagenome_%j.err

# Usage examples:
#   sbatch scripts/slurm/train_student_alphagenome.sh
#   sbatch --export=ALL,CFG=student_alphagenome_k562,HEAD=pool-flatten,SUBSET=0.25 scripts/slurm/train_student_alphagenome.sh

source /etc/profile.d/modules.sh
module load EB5

cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
mkdir -p logs

CFG="${CFG:-student_alphagenome_yeast}"
HEAD="${HEAD:-mlp-512-512}"
WANDB_MODE="${WANDB_MODE:-offline}"
SUBSET="${SUBSET:-null}"
WEIGHTS_PATH="${ALPHAGENOME_WEIGHTS_PATH:-checkpoints/alphagenome-jax-all_folds-v1}"

if [ "$SUBSET" = "null" ]; then
  ~/.local/bin/uv run --no-sync python experiments/train_oracle_alphagenome.py \
    --config-name "${CFG}" \
    head_arch="${HEAD}" \
    weights_path="${WEIGHTS_PATH}" \
    wandb_mode="${WANDB_MODE}"
else
  ~/.local/bin/uv run --no-sync python experiments/train_oracle_alphagenome.py \
    --config-name "${CFG}" \
    head_arch="${HEAD}" \
    subset_fraction="${SUBSET}" \
    weights_path="${WEIGHTS_PATH}" \
    wandb_mode="${WANDB_MODE}"
fi
