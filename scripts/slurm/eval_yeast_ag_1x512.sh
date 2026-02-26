#!/bin/bash
#SBATCH --job-name=eval_ag1x512
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpuq
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --constraint=h100
#SBATCH --time=2:00:00

source /etc/profile.d/modules.sh
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

# We evaluate the flatten_1x512 model 
uv run python scripts/analysis/eval_yeast_ag.py outputs/ag_yeast_sweep/flatten_1x512/best_model ag_yeast_flatten_mlp_v1 flatten-mlp-512
