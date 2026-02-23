#!/bin/bash
#SBATCH --job-name=albench_exp0
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=kooq
#SBATCH --qos=koolab
#SBATCH --nodelist=bamgpu101
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=14
#SBATCH --mem=200G
#SBATCH --time=48:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=christen@cshl.edu

cd /grid/koo/data/christen/ALBench-S2F || exit 1
module load EB5
uv sync
export WANDB_API_KEY=$(cat ~/.wandb_key)
uv run python experiments/exp0_scaling.py +task=k562 +student=dream_rnn experiment.dry_run=false
cp -r "outputs" /grid/koo/data/christen/ALBench-S2F/outputs/
