#!/bin/bash
# Generate LegNet oracle pseudo-labels for K562 (10-fold cross-validation).
# Uses quality-filtered hashfrag splits (hashfrag_splits_qf/).
#
# Submit:
#   /cm/shared/apps/slurm/current/bin/sbatch scripts/slurm/legnet_oracle_k562.sh
#
#SBATCH --job-name=ln_oracle
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00

set -euo pipefail
set +u; source /etc/profile.d/modules.sh; set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
source scripts/slurm/setup_hpc_deps.sh

echo "=== LegNet oracle K562 (10-fold) — $(date) ==="
echo "Node: ${SLURMD_NODENAME}"

uv run --no-sync python experiments/generate_oracle_pseudolabels_k562_legnet.py

echo "=== Done — $(date) ==="
