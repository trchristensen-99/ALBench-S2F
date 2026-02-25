#!/bin/bash
#SBATCH --job-name=ev_pool
#SBATCH --partition=gpuq
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00

source .venv/bin/activate
uv run python eval_ag.py outputs/oracle_alphagenome_k562_pool/last_model alphagenome_k562_pool_head
