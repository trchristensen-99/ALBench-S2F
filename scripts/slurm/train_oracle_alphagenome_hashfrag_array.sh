#!/bin/bash
# Train AlphaGenome hashFrag oracle — 10 independent seeds (42–51).
# Best config: boda-flatten-512-512, dropout=0.1, lr=0.001, full RC+shift±15bp aug.
# Submit: sbatch train_oracle_alphagenome_hashfrag_array.sh
#
# Each job writes to outputs/ag_hashfrag_oracle/seed_${SEED}/
#   best_model/   — best checkpoint by val Pearson R
#   test_metrics.json — in_distribution / snv_abs / snv_delta / ood results

#SBATCH --job-name=ag_hf_oracle
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=200G
#SBATCH --array=0-9

SEEDS=(42 43 44 45 46 47 48 49 50 51)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

source /etc/profile.d/modules.sh
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD:$PYTHONPATH"
source scripts/slurm/setup_hpc_deps.sh

# Prevent XLA command buffer compilation issue on H100
export XLA_FLAGS="${XLA_FLAGS} --xla_gpu_enable_command_buffer="

echo "Starting seed ${SEED} (array task ${SLURM_ARRAY_TASK_ID})"

uv run python experiments/train_oracle_alphagenome_hashfrag.py \
    ++seed=${SEED} \
    ++output_dir=outputs/ag_hashfrag_oracle/seed_${SEED} \
    ++wandb_mode=offline
