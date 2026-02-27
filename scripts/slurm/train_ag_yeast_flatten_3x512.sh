#!/bin/bash
#SBATCH --job-name=ag_yeast_f3x512
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpuq
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
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

uv run python experiments/train_oracle_alphagenome_yeast.py \
    ++head_arch="flatten-mlp" \
    "++hidden_dims=[512,512,512]" \
    ++dropout_rate=0.1 \
    ++lr=1e-4 \
    ++weight_decay=1e-4 \
    ++lr_schedule=plateau \
    ++aug_mode=no_shift \
    ++output_dir=outputs/ag_yeast_sweep/flatten_3x512 \
    ++cache_dir=outputs/ag_yeast_sweep/final_embedding_cache \
    ++seed=42 \
    ++wandb_mode=online \
    ++batch_size=4096
