#!/bin/bash
#SBATCH --job-name=eval_boda_k562
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpuq
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1

source /etc/profile.d/modules.sh
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1

export PYTHONPATH="$PWD:$PYTHONPATH"

uv run python scripts/analysis/eval_boda_k562.py

