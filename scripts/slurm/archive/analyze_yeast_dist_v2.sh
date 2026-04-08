#!/bin/bash
# Run yeast oracle distribution analysis on v2 pseudolabels.
# Submit with: --dependency=afterok:<pseudolabel_job_id>
#
#SBATCH --job-name=yeast_dist_v2
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

echo "=== Yeast oracle distribution analysis (v2 pseudolabels) ==="
echo "Date: $(date)"

uv run --no-sync python scripts/analysis/analyze_yeast_oracle_label_distributions.py \
    --pseudolabel-dir outputs/oracle_pseudolabels/yeast_dream_oracle_v2 \
    --out-dir outputs/analysis/yeast_oracle_label_distributions_v2

echo "=== DONE — $(date) ==="
