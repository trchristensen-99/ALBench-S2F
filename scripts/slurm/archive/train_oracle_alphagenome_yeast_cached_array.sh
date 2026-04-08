#!/bin/bash
# Train 10-seed AlphaGenome yeast oracle in cached no_shift mode.
# Requires shared cache at outputs/ag_yeast/embedding_cache.
#
# Submit:
#   /cm/shared/apps/slurm/current/bin/sbatch scripts/slurm/train_oracle_alphagenome_yeast_cached_array.sh
#
#SBATCH --job-name=ag_yeast_oracle
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --time=48:00:00
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


echo "Starting yeast AG oracle task=${SLURM_ARRAY_TASK_ID}"
echo "Node: ${SLURMD_NODENAME}  Date: $(date)"

uv run --no-sync python experiments/train_oracle_alphagenome_yeast.py \
    --config-name oracle_alphagenome_yeast \
    ++cache_dir=outputs/ag_yeast/embedding_cache \
    ++output_dir="outputs/ag_yeast_oracle_cached/oracle_${SLURM_ARRAY_TASK_ID}" \
    ++wandb_mode=offline
