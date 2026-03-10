#!/bin/bash
# Re-evaluate ALL DREAM-RNN scaling results with correct OOD test set.
# The old runs evaluated against test_ood_cre.tsv (wrong); this backfill
# uses test_ood_designed_k562.tsv (correct, N=22,862).
#
# Submit:
#   /cm/shared/apps/slurm/current/bin/sbatch scripts/slurm/backfill_dreamrnn_ood.sh

#SBATCH --job-name=backfill_dreamrnn_ood
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpuq
#SBATCH --qos=fast
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G

set -euo pipefail

set +u
source /etc/profile.d/modules.sh
set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
source scripts/slurm/setup_hpc_deps.sh

echo "Backfilling DREAM-RNN K562 test metrics — $(date)"
echo "Node: ${SLURMD_NODENAME}"

DATA=data/k562

for seed_dir in outputs/exp0_k562_scaling/seed_*; do
    if [ ! -d "$seed_dir" ]; then
        continue
    fi
    # Skip seeds with no result.json (incomplete or failed runs)
    if ! ls "$seed_dir"/fraction_*/result.json &>/dev/null; then
        echo "=== Skipping $seed_dir (no result.json) ==="
        continue
    fi
    echo "=== Processing $seed_dir ==="
    uv run --no-sync python scripts/analysis/backfill_exp0_k562_test_metrics.py \
        --output-root "$seed_dir" \
        --data-path "$DATA" \
        --device cuda:0 \
        --batch-size 256 \
        --force
done

echo "Done — $(date)"
