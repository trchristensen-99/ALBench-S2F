#!/bin/bash
#SBATCH --job-name=ag_cache_test
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpuq
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1


source /etc/profile.d/modules.sh
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1

export PYTHONPATH="$PWD:$PYTHONPATH"

# Test: flatten head (tests v4 rename fix) + no_shift cache mode (tests embedding cache).
# Cache is built on full ds_train (~700k seqs) once, then training uses head-only steps.
# Expected total time: ~35-45 min (cache build) + minutes (3 epochs head-only).
uv run python experiments/train_oracle_alphagenome_full.py \
    ++head_arch="boda-flatten-512-512" \
    ++aug_mode="no_shift" \
    ++epochs=3 \
    ++gpu=0 \
    ++seed=42 \
    ++output_dir=outputs/ag_cache_test
