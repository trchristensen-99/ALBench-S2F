#!/bin/bash
#SBATCH --job-name=albench_exp0
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=christen@cshl.edu

set -euo pipefail

cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1

export PYTHONPATH="$PWD:$PYTHONPATH"

# Ensure logs directory exists
mkdir -p logs

# Bootstrap environment
bash scripts/setup_runtime.sh

# W&B authentication
if [[ -f ~/.wandb_key ]]; then
    export WANDB_API_KEY=$(cat ~/.wandb_key)
elif [[ -f .env ]]; then
    export $(grep WANDB_API_KEY .env | xargs)
fi

# Run experiment (override args via command line)
bash scripts/run_with_runtime.sh python experiments/exp0_scaling.py \
    +task=k562 +student=dream_rnn \
    experiment.dry_run=false \
    "$@"
