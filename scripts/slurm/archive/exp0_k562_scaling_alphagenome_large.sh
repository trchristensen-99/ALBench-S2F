#!/bin/bash
# Exp 0: AlphaGenome head scaling curve on K562 — large fractions only.
# Runs fractions 0.50 and 1.00 which need >12h to complete.
# Each array task runs one fraction independently.
# Submit: sbatch exp0_k562_scaling_alphagenome_large.sh
#
#SBATCH --job-name=exp0_ag_k562
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=200G
#SBATCH --array=0-1

source /etc/profile.d/modules.sh
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD:$PYTHONPATH"
source scripts/slurm/setup_hpc_deps.sh

export XLA_FLAGS="${XLA_FLAGS} --xla_gpu_enable_command_buffer= --xla_gpu_autotune_level=0"

FRACTIONS=(0.50 1.00)
FRACTION=${FRACTIONS[$SLURM_ARRAY_TASK_ID]}

echo "Starting AG scaling (large): fraction=${FRACTION} (task ${SLURM_ARRAY_TASK_ID}/1)"
echo "Node: $SLURMD_NODENAME  Date: $(date)"

uv run --no-sync python experiments/exp0_k562_scaling_alphagenome.py \
    ++fraction="${FRACTION}" \
    ++wandb_mode=offline
