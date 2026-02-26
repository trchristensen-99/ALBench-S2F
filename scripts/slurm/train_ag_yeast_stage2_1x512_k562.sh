#!/bin/bash
#SBATCH --job-name=ag_yeast_s2_1x512
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
mkdir -p logs

# Use MPRA K562 best practices: batch size 32, LR 1e-5, weight decay 1e-6
# Skip Stage 1 via epochs=0 since we load the pretrained head directly
uv run python experiments/train_oracle_alphagenome_yeast.py \
    ++head_arch="flatten-mlp" \
    "++hidden_dims=[512]" \
    ++dropout_rate=0.1 \
    ++epochs=0 \
    ++pretrained_head_dir=outputs/ag_yeast_sweep/flatten_1x512/best_model \
    ++batch_size=4096 \
    ++second_stage_lr=1e-5 \
    ++second_stage_epochs=50 \
    ++second_stage_batch_size=32 \
    ++second_stage_weight_decay=1e-6 \
    ++aug_mode=no_shift \
    ++output_dir=outputs/ag_yeast_stage2/flatten_1x512_k562 \
    ++cache_dir=outputs/ag_yeast_sweep/final_embedding_cache \
    ++seed=42 \
    ++wandb_mode=online
