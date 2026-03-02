#!/bin/bash
#SBATCH --job-name=exp0_yeast
#SBATCH --output=logs/exp0_yeast_%a.out
#SBATCH --error=logs/exp0_yeast_%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=200G
#SBATCH --time=48:00:00
#SBATCH --array=0-29

set -euo pipefail

# Source system profiles for modules and tmpdir
set +u
source /etc/profile.d/modules.sh
set -u
module load EB5

cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1

export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p logs
source scripts/slurm/setup_hpc_deps.sh

# Map array index to (seed, fraction)
FRACTIONS=(0.001 0.002 0.005 0.01 0.02 0.05 0.1 0.2 0.5 1.0)
SEEDS=(42 43 44)
N_FRAC=${#FRACTIONS[@]}
FRAC_IDX=$((SLURM_ARRAY_TASK_ID % N_FRAC))
SEED_IDX=$((SLURM_ARRAY_TASK_ID / N_FRAC))
FRACTION=${FRACTIONS[$FRAC_IDX]}
SEED=${SEEDS[$SEED_IDX]}

echo "=== Exp 0 Yeast Scaling: seed=${SEED} fraction=${FRACTION} task=${SLURM_ARRAY_TASK_ID} ==="

# W&B auth
if [[ -f ~/.wandb_key ]]; then
    export WANDB_API_KEY=$(cat ~/.wandb_key)
elif [[ -f .env ]]; then
    export $(grep WANDB_API_KEY .env | xargs)
fi

uv run --no-sync python experiments/exp0_yeast_scaling.py \
    fraction="${FRACTION}" \
    output_dir=outputs/exp0_yeast_scaling \
    seed="${SEED}" \
    test_subset_dir=data/yeast/test_subset_ids \
    wandb_mode=offline
