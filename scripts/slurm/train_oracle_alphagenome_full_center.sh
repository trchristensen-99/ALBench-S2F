#!/bin/bash
#SBATCH --job-name=ag_orc_full_center
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpuq
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1


source /etc/profile.d/modules.sh
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1

export PYTHONPATH="$PWD:$PYTHONPATH"

# no_shift for fast iteration. For production use ++aug_mode="full".
uv run python experiments/train_oracle_alphagenome_full.py \
    ++head_arch="boda-center-512-512" \
    ++aug_mode="no_shift" \
    ++gpu=0 \
    ++seed=42 \
    ++output_dir=outputs/ag_center

# cp -r outputs/ag_center /grid/wsbs/home_norepl/christen/ALBench-S2F/outputs/
