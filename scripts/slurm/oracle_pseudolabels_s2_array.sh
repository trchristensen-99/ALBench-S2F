#!/bin/bash
# Stage 2 oracle pseudolabel generation — PARALLELIZED per-fold array job.
# Each task processes one fold independently. 10x faster than sequential.
#
# Submit:
#   ARRAY_JOB=$(/cm/shared/apps/slurm/current/bin/sbatch --parsable scripts/slurm/oracle_pseudolabels_s2_array.sh)
#   /cm/shared/apps/slurm/current/bin/sbatch --dependency=afterok:$ARRAY_JOB scripts/slurm/aggregate_oracle_pseudolabels_s2.sh
#
#SBATCH --job-name=oracle_pl_s2_fold
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --time=12:00:00
#SBATCH --requeue
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

export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_command_buffer= --xla_gpu_autotune_level=0"
export PYTHONUNBUFFERED=1

FOLD_ID=${SLURM_ARRAY_TASK_ID}
echo "=== S2 pseudolabel gen: fold ${FOLD_ID} ==="
echo "Node: ${SLURMD_NODENAME}  Date: $(date)"
START_TIME=$(date +%s)

uv run --no-sync python experiments/generate_stage2_pseudolabels_single_fold.py \
    ++fold_id="${FOLD_ID}" \
    ++wandb_mode=disabled

END_TIME=$(date +%s)
echo "=== Fold ${FOLD_ID} DONE — wall time: $((END_TIME - START_TIME))s ==="
