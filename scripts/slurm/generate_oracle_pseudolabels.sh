#!/bin/bash
# Generate oracle pseudo-labels for all K562 hashFrag sequences.
# Loads all 10 oracle checkpoints and runs ensemble RC-averaged inference
# on train+pool (~320K), val (~41K), and all test sets.
#
# Submit:
#   /cm/shared/apps/slurm/current/bin/sbatch scripts/slurm/generate_oracle_pseudolabels.sh
#
#SBATCH --job-name=oracle_pseudolabels
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpuq
#SBATCH --qos=fast
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G

set -euo pipefail

set +u
source /etc/profile.d/modules.sh
set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
source scripts/slurm/setup_hpc_deps.sh

export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_command_buffer= --xla_gpu_autotune_level=0"

echo "Generating oracle pseudo-labels — $(date)"
echo "Node: ${SLURMD_NODENAME}"

uv run --no-sync python experiments/generate_oracle_pseudolabels.py \
    ++wandb_mode=disabled
