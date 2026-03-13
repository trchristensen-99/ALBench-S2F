#!/bin/bash
# Prepare oracle-labeled test sets for Experiment 1.
# Requires AG oracle checkpoints and yeast DREAM oracle on HPC.
#
#SBATCH --job-name=prep_exp1_test
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=200G

set -euo pipefail

set +u
source /etc/profile.d/modules.sh
set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
source scripts/slurm/setup_hpc_deps.sh

export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_command_buffer="

echo "=== Preparing Experiment 1 test sets ==="
echo "Node: $SLURMD_NODENAME  Date: $(date)"

uv run --no-sync python scripts/prepare_exp1_test_sets.py

echo "Done: $(date)"
