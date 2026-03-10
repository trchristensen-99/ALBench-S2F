#!/bin/bash
# Evaluate 384bp compact-window heads on chr 7, 13 test set.
# Builds the compact test embedding cache (T=3) if not already present,
# then runs eval for all compact configs in eval_ag_chrom_test.py.
#
# Usage (from HPC repo root):
#   sbatch scripts/slurm/eval_ag_compact.sh
#SBATCH --job-name=ag_compact_eval
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

COMPACT_CACHE="outputs/ag_compact/embedding_cache_compact"

# Step 1: Build 384bp compact test embedding cache (idempotent — skips if already done).
echo "[eval_ag_compact] Building compact test cache (384bp, T=3)..."
uv run python scripts/analysis/build_test_embedding_cache.py \
    --cache_dir "$COMPACT_CACHE" \
    --seq_len 384

# Step 2: Evaluate all compact heads (sum/mean/max/center) on chr 7, 13.
# Compact configs in eval_ag_chrom_test.py use $COMPACT_CACHE automatically.
# The --cache_dir arg here covers 600bp no_shift/hybrid heads; compact heads override it.
echo "[eval_ag_compact] Evaluating compact heads on chr 7, 13..."
uv run python scripts/analysis/eval_ag_chrom_test.py \
    --data_path data/k562 \
    --cache_dir outputs/ag_flatten/embedding_cache \
    --output outputs/ag_chrom_test_results.json
