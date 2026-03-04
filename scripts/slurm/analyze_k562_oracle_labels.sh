#!/bin/bash
# K562 oracle label distribution analysis (CPU-only, lightweight).
#
# Submit:
#   /cm/shared/apps/slurm/current/bin/sbatch scripts/slurm/analyze_k562_oracle_labels.sh
#
#SBATCH --job-name=analyze_k562
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpuq
#SBATCH --qos=fast
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -euo pipefail

set +u
source /etc/profile.d/modules.sh
set -u
module load EB5

cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
source scripts/slurm/setup_hpc_deps.sh

echo "Running K562 oracle label distribution analysis at $(date)"
uv run --no-sync python scripts/analysis/analyze_k562_oracle_label_distributions.py
echo "Done at $(date)"
