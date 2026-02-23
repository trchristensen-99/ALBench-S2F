#!/bin/bash
# Fast AlphaGenome head run for rapid iteration: no_shift cache + 5 epochs + 2h.
# Use for sanity checks and architecture search. For production, use the full_* scripts with ++aug_mode="full".
#SBATCH --job-name=ag_fast
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

# boda-flatten by default; override with head_arch (e.g. boda-sum-512-512) if desired.
uv run python experiments/train_oracle_alphagenome_full.py \
    ++head_arch="boda-flatten-512-512" \
    ++aug_mode="no_shift" \
    ++epochs=5 \
    ++batch_size=256 \
    ++gpu=0 \
    ++seed=42 \
    ++output_dir=outputs/ag_fast
