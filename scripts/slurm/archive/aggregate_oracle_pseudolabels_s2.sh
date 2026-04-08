#!/bin/bash
# Aggregate per-fold Stage 2 pseudolabel predictions into final NPZ files.
# Submit with: --dependency=afterok:<array_job_id>
#
#SBATCH --job-name=agg_oracle_pl_s2
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

echo "=== Aggregating S2 oracle pseudolabels — $(date) ==="

uv run --no-sync python scripts/analysis/aggregate_stage2_oracle_pseudolabels.py \
    --preds-dir outputs/oracle_pseudolabels_stage2_k562_ag/fold_preds \
    --output-dir outputs/oracle_pseudolabels_stage2_k562_ag \
    --k562-data-path data/k562

echo "=== DONE — $(date) ==="
