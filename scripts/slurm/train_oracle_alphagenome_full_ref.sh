#!/bin/bash
#SBATCH --job-name=ag_orc_full_ref
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpuq
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1


source /etc/profile.d/modules.sh
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1

export PYTHONPATH="$PWD:$PYTHONPATH"

uv run python experiments/train_oracle_alphagenome_full.py \
    ++head_arch="encoder-1024-dropout" \
    ++gpu=0 \
    ++seed=42 \
    ++output_dir=outputs/ag_ref

# cp -r outputs/ag_ref /grid/wsbs/home_norepl/christen/ALBench-S2F/outputs/
