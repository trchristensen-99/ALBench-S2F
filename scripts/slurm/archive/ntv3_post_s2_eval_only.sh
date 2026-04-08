#!/bin/bash
# Evaluate NTv3-Borzoi S2 checkpoints without retraining.
# Loads best_encoder_state.pkl + best_head.pt and runs test eval only.
# Safe to run while training jobs are still running (read-only on checkpoints).
#
# Submit:
#   /cm/shared/apps/slurm/current/bin/sbatch scripts/slurm/ntv3_post_s2_eval_only.sh
#
#SBATCH --job-name=ntv3p_eval
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpuq
#SBATCH --qos=fast
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00

set -euo pipefail

set +u
source /etc/profile.d/modules.sh
set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
source scripts/slurm/setup_hpc_deps.sh

echo "=== NTv3-Borzoi S2 Eval-Only ==="
echo "Node: ${SLURMD_NODENAME}  Date: $(date)"

uv run --no-sync python scripts/eval_ntv3_post_s2.py

echo "DONE — $(date)"
