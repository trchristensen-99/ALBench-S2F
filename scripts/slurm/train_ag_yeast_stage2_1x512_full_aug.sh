#!/bin/bash
#SBATCH --job-name=ag_yeast_s2_1x512_full
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpuq
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --constraint=h100
#SBATCH --time=12:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=christen@cshl.edu

source /etc/profile.d/modules.sh
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD:$PYTHONPATH"

# Ensure CUDA-enabled JAX and HPC-specific AlphaGenome dependencies are loaded
source scripts/slurm/setup_hpc_deps.sh

mkdir -p logs

# Full sequence augmentation + lower LR for maximum generalisation robustness
uv run python experiments/train_oracle_alphagenome_yeast.py \
    ++head_arch="flatten-mlp" \
    "++hidden_dims=[512]" \
    ++dropout_rate=0.1 \
    ++epochs=0 \
    ++pretrained_head_dir=outputs/ag_yeast_sweep/flatten_1x512/best_model \
    ++batch_size=4096 \
    ++second_stage_lr=5e-6 \
    ++second_stage_epochs=50 \
    ++second_stage_batch_size=32 \
    ++second_stage_weight_decay=1e-6 \
    ++aug_mode=full \
    ++output_dir=outputs/ag_yeast_stage2/flatten_1x512_full_aug \
    ++cache_dir=outputs/ag_yeast_sweep/final_embedding_cache \
    ++seed=42 \
    ++wandb_mode=online
