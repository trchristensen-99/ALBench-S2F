#!/bin/bash
# Evaluate v2/ref heads (dropout + LR sweep) on chr 7, 13 test set.
# Skips any configs whose checkpoints are not yet present.
# The 600bp test embedding cache must already exist at
# outputs/ag_flatten/embedding_cache (built during original ag_flatten run).
#SBATCH --job-name=ag_v2_eval
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpuq
#SBATCH --time=02:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1

set -e
source /etc/profile.d/modules.sh
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD:$PYTHONPATH"
source scripts/slurm/setup_hpc_deps.sh

echo "[eval_ag_v2] Evaluating v2/ref heads on chr 7, 13 test set..."
uv run python scripts/analysis/eval_ag_chrom_test.py \
    --data_path data/k562 \
    --cache_dir outputs/ag_flatten/embedding_cache \
    --output outputs/ag_chrom_test_results_v2.json

echo "[eval_ag_v2] Done. Results at outputs/ag_chrom_test_results_v2.json"
