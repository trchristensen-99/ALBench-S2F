#!/bin/bash
#SBATCH --job-name=exp0_yeast
#SBATCH --output=logs/exp0_yeast_%a.out
#SBATCH --error=logs/exp0_yeast_%a.err
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --array=0-9
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=christen@cshl.edu

set -euo pipefail

# Source system profiles for modules and tmpdir
source /etc/profile.d/modules.sh
source /etc/profile.d/slurm_tmpdir.sh
module load EB5

cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
mkdir -p logs

# Map array index to fraction
FRACTIONS=(0.001 0.002 0.005 0.01 0.02 0.05 0.1 0.2 0.5 1.0)
FRACTION=${FRACTIONS[$SLURM_ARRAY_TASK_ID]}

echo "=== Exp 0 Yeast Scaling: fraction=${FRACTION} (task ${SLURM_ARRAY_TASK_ID}) ==="

# W&B auth
if [[ -f ~/.wandb_key ]]; then
    export WANDB_API_KEY=$(cat ~/.wandb_key)
elif [[ -f .env ]]; then
    export $(grep WANDB_API_KEY .env | xargs)
fi

uv run python experiments/exp0_yeast_scaling.py \
    --fraction "${FRACTION}" \
    --seed 42 \
    --wandb-mode offline
