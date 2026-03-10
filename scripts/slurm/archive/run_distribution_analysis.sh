#!/bin/bash
# Run oracle label distribution analysis for both K562 and yeast.
# CPU-only job — no GPU needed (just loads npz files and makes plots).
#
# Submit:
#   /cm/shared/apps/slurm/current/bin/sbatch scripts/slurm/run_distribution_analysis.sh
#
#SBATCH --job-name=dist_analysis
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G

set -euo pipefail

set +u
source /etc/profile.d/modules.sh
set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
source scripts/slurm/setup_hpc_deps.sh

echo "=== K562 oracle label distribution analysis ==="
echo "Started at $(date)"
uv run --no-sync python scripts/analysis/analyze_k562_oracle_label_distributions.py
echo "K562 analysis done at $(date)"

echo ""
echo "=== Yeast oracle label distribution analysis ==="
echo "Started at $(date)"
uv run --no-sync python scripts/analysis/analyze_yeast_oracle_label_distributions.py
echo "Yeast analysis done at $(date)"
