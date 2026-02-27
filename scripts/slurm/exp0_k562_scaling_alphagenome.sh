#!/bin/bash
# Exp 0: AlphaGenome head scaling curve on K562.
# Trains boda-flatten-512-512 on 7 random downsamples of hashFrag train+pool.
# Each array task runs one fraction independently.
# Submit: sbatch exp0_k562_scaling_alphagenome.sh
#
#SBATCH --job-name=exp0_ag_k562
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=200G
#SBATCH --array=0-6

source /etc/profile.d/modules.sh
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD:$PYTHONPATH"
source scripts/slurm/setup_hpc_deps.sh

export XLA_FLAGS="${XLA_FLAGS} --xla_gpu_enable_command_buffer= --xla_gpu_autotune_level=0"

FRACTIONS=(0.01 0.02 0.05 0.10 0.20 0.50 1.00)
FRACTION=${FRACTIONS[$SLURM_ARRAY_TASK_ID]}

echo "Starting AG scaling: fraction=${FRACTION} (task ${SLURM_ARRAY_TASK_ID}/6)"
echo "Node: $SLURMD_NODENAME  Date: $(date)"

uv run python experiments/exp0_k562_scaling_alphagenome.py \
    ++fraction="${FRACTION}" \
    ++seed=null \
    ++wandb_mode=offline
