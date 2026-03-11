#!/bin/bash
# K562 oracle distribution analysis using Stage 2 pseudolabels.
# Submit with: --dependency=afterok:<pseudolabel_job_id>
#
#SBATCH --job-name=k562_dist_s2
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

echo "=== K562 oracle distribution analysis (S2 pseudolabels) ==="
echo "Date: $(date)"

uv run --no-sync python scripts/analysis/analyze_k562_oracle_label_distributions.py \
    --pseudolabel-dir outputs/oracle_pseudolabels_stage2_k562_ag \
    --out-dir outputs/analysis/k562_oracle_label_distributions_s2

echo "=== DONE — $(date) ==="
