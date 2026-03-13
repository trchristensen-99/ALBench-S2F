#!/bin/bash
# Generate yeast AlphaGenome oracle pseudolabels for Experiment 1 test sets.
# Run AFTER train_oracle_alphagenome_yeast_ensemble.sh completes.
#
# Creates: data/yeast/test_sets_ag/ with oracle-labeled NPZ files
#
#SBATCH --job-name=yeast_ag_plabels
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

echo "=== Generating yeast AG oracle pseudolabels ==="
echo "Node: $SLURMD_NODENAME  Date: $(date)"

uv run --no-sync python experiments/generate_oracle_pseudolabels_yeast_ag.py

echo "Done: $(date)"
