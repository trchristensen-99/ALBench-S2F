#!/bin/bash
# Train yeast AlphaGenome oracle ensemble (10 folds).
# Each fold trains a cached head-only model (S1) with different seed.
# Outputs: outputs/oracle_alphagenome_yeast_ensemble/oracle_N/
#
#SBATCH --job-name=oracle_ag_yeast
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=200G
#SBATCH --array=0-9

set -euo pipefail

set +u
source /etc/profile.d/modules.sh
set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
source scripts/slurm/setup_hpc_deps.sh

export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_command_buffer="

FOLD=$SLURM_ARRAY_TASK_ID
SEED=$((42 + FOLD * 7919))

echo "=== Yeast AG Oracle: fold=${FOLD}, seed=${SEED} ==="
echo "Node: $SLURMD_NODENAME  Date: $(date)"

uv run --no-sync python experiments/train_oracle_alphagenome_yeast.py \
    --config-name oracle_alphagenome_yeast_cached \
    ++output_dir="outputs/oracle_alphagenome_yeast_ensemble/oracle_${FOLD}" \
    ++seed="${SEED}" \
    ++wandb_mode=offline

echo "Done: $(date)"
