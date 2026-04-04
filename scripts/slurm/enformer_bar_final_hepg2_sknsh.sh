#!/bin/bash
#SBATCH --job-name=enformer_bar_hepg2_sknsh
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

# Build Enformer embedding caches for HepG2 and SknSh, then train S1 heads (3 seeds each)
# Cache takes ~10h per cell type, so this needs 48h total

for CELL in hepg2 sknsh; do
    echo "=== Building Enformer embedding cache for $CELL ==="
    CACHE_DIR="outputs/bar_final/${CELL}/enformer_cached/embedding_cache"
    OUTPUT_DIR="outputs/bar_final/${CELL}/enformer_cached"

    uv run --no-sync python scripts/build_enformer_embedding_cache.py \
        --data-path data/k562 \
        --cache-dir "$CACHE_DIR" \
        --chr-split \
        --include-alt-alleles \
        --splits train val \
        --include-test \
        --batch-size 4

    echo "=== Training Enformer S1 heads for $CELL (3 seeds) ==="
    for SEED in 42 1042 2042; do
        echo "--- Seed $SEED ---"
        uv run --no-sync python experiments/train_foundation_cached.py \
            ++model_name=enformer \
            ++cache_dir="$CACHE_DIR" \
            ++embed_dim=3072 \
            ++output_dir="${OUTPUT_DIR}/seed_${SEED}" \
            ++cell_line="$CELL" \
            ++chr_split=True \
            ++include_alt_alleles=True \
            ++seed="$SEED" \
            ++lr=0.001 \
            ++dropout=0.1 \
            ++weight_decay=1e-6 \
            ++early_stop_patience=10
    done
done
