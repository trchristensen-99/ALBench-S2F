#!/bin/bash
# Build Enformer embedding cache for bar_final (chr-split, ref+alt).
# K562 only — HepG2/SknSh share the same sequences (just different labels).
# ~10h for train (618K seqs at ~4 it/s), plus val/test.
#
# Submit:
#   /cm/shared/apps/slurm/current/bin/sbatch scripts/slurm/enformer_cache_bar_final.sh
#
#SBATCH --job-name=enf_cache2
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=48:00:00

set -euo pipefail

set +u
source /etc/profile.d/modules.sh
set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
source scripts/slurm/setup_hpc_deps.sh

CACHE_DIR="outputs/bar_final/k562/enformer_cached/embedding_cache"

echo "=== Enformer cache for bar_final (chr-split, ref+alt) ==="
echo "Node: ${SLURMD_NODENAME} — $(date)"

# Check what's already done
NEED_BUILD=""
for f in train_canonical.npy val_canonical.npy test_in_dist_canonical.npy test_snv_ref_canonical.npy test_snv_alt_canonical.npy test_ood_canonical.npy; do
    if [ ! -f "${CACHE_DIR}/$f" ]; then
        NEED_BUILD="yes"
        echo "Missing: $f"
    fi
done

if [ -z "$NEED_BUILD" ]; then
    echo "All cache files exist. Skipping cache build."
else
    uv run --no-sync python scripts/build_enformer_embedding_cache.py \
        --data-path data/k562 \
        --cache-dir "$CACHE_DIR" \
        --chr-split \
        --include-alt-alleles \
        --splits train val \
        --include-test \
        --batch-size 4
fi

echo "=== Enformer cache DONE — $(date) ==="

# Now train S1 heads for all 3 cell types (quick, ~5-10 min each)
echo ""
echo "=== Training Enformer S1 heads (3 cell types) ==="

for CELL in k562 hepg2 sknsh; do
    OUT="outputs/bar_final/${CELL}/enformer_s1_v2"
    if [ -f "${OUT}/seed_42/seed_42/result.json" ]; then
        echo "  ${CELL}: already done"
        continue
    fi
    echo "  Training ${CELL}..."
    uv run --no-sync python experiments/train_foundation_cached.py \
        --model enformer \
        --cache-dir "$CACHE_DIR" \
        --output-dir "$OUT" \
        --cell-line "$CELL" \
        --chr-split \
        --include-alt-alleles \
        --seeds 42 123 456 \
        --lr 0.001 \
        --dropout 0.1 \
        --weight-decay 1e-6 \
        --patience 10
done

echo "=== All done — $(date) ==="
