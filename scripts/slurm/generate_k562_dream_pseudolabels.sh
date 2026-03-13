#!/bin/bash
# Generate K562 DREAM-RNN oracle pseudolabels for Experiment 1 test sets.
# Run AFTER train_oracle_dream_rnn_k562_ensemble.sh completes.
#
#SBATCH --job-name=k562_dream_plabels
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

set -euo pipefail

set +u
source /etc/profile.d/modules.sh
set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
source scripts/slurm/setup_hpc_deps.sh

echo "=== Generating K562 DREAM-RNN oracle pseudolabels ==="
echo "Node: $SLURMD_NODENAME  Date: $(date)"

uv run --no-sync python experiments/generate_oracle_pseudolabels_k562_dream.py

echo "Done: $(date)"
