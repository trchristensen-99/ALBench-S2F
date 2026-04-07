#!/bin/bash
# Rebuild foundation model chr_split embedding caches with correct test alignment.
#
# Problem: Existing v2 caches in outputs/chr_split/{cell}/{model}_cached_v2/ were
# built with --include-alt-alleles (Enformer) or mismatched test splits, creating
# a length mismatch (33K or 40K embeddings vs 31K chr-split test labels).
#
# Fix: Rebuild caches with --chr-split and WITHOUT --include-alt-alleles so the
# test set has exactly ~31,435 sequences matching the chr-split labels.
#
# Only rebuilds TEST embeddings (train embeddings are not needed for scatter plots).
# Saves to outputs/chr_split/{cell}/{model}_cached_v3/embedding_cache/ (v3).
#
# Array (9 tasks):
#   0: Enformer K562    1: Borzoi K562    2: NTv3 K562
#   3: Enformer HepG2   4: Borzoi HepG2   5: NTv3 HepG2
#   6: Enformer SknSh   7: Borzoi SknSh   8: NTv3 SknSh
#
# Submit:
#   /cm/shared/apps/slurm/current/bin/sbatch --array=0-8 scripts/slurm/rebuild_foundation_chr_split_caches.sh
#
# Or split across QoS tiers for faster scheduling:
#   /cm/shared/apps/slurm/current/bin/sbatch --array=0,3,6 --qos=fast scripts/slurm/rebuild_foundation_chr_split_caches.sh   # Enformer (smaller)
#   /cm/shared/apps/slurm/current/bin/sbatch --array=2,5,8 --qos=fast scripts/slurm/rebuild_foundation_chr_split_caches.sh   # NTv3 (fast)
#   /cm/shared/apps/slurm/current/bin/sbatch --array=1,4,7 --qos=default scripts/slurm/rebuild_foundation_chr_split_caches.sh # Borzoi (large model)
#
#SBATCH --job-name=rebuild_cache
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=fast
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=200G

set -euo pipefail

set +u; source /etc/profile.d/modules.sh; set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
source scripts/slurm/setup_hpc_deps.sh

T=$SLURM_ARRAY_TASK_ID
echo "=== rebuild_foundation_chr_split_caches task=${T} node=${SLURMD_NODENAME} date=$(date) ==="

# Model and cell arrays (3 models x 3 cells = 9 tasks)
MODELS=("enformer" "borzoi" "ntv3_post" "enformer" "borzoi" "ntv3_post" "enformer" "borzoi" "ntv3_post")
CELLS=("k562" "k562" "k562" "hepg2" "hepg2" "hepg2" "sknsh" "sknsh" "sknsh")

MODEL="${MODELS[$T]}"
CELL="${CELLS[$T]}"
CACHE_DIR="outputs/chr_split/${CELL}/${MODEL}_cached_v3/embedding_cache"

echo "Model: ${MODEL}, Cell: ${CELL}"
echo "Cache: ${CACHE_DIR}"

# Ensure data symlinks exist for HepG2/SknSh (they share the same data file)
if [ "${CELL}" != "k562" ]; then
    mkdir -p "data/${CELL}"
    ln -sf "$(pwd)/data/k562/DATA-Table_S2__MPRA_dataset.txt" "data/${CELL}/DATA-Table_S2__MPRA_dataset.txt" 2>/dev/null || true
    ln -sf "$(pwd)/data/k562/hashfrag_splits" "data/${CELL}/hashfrag_splits" 2>/dev/null || true
    ln -sf "$(pwd)/data/k562/test_sets" "data/${CELL}/test_sets" 2>/dev/null || true
fi

# Determine build script and extra flags per model
case ${MODEL} in
    enformer)
        BUILD_SCRIPT="scripts/build_enformer_embedding_cache.py"
        EXTRA_FLAGS=""
        ;;
    borzoi)
        BUILD_SCRIPT="scripts/build_borzoi_embedding_cache.py"
        # Borzoi needs --no-autocast for float32 stability
        EXTRA_FLAGS="--no-autocast"
        ;;
    ntv3_post)
        BUILD_SCRIPT="scripts/build_ntv3_embedding_cache.py"
        EXTRA_FLAGS="--model-variant post"
        ;;
esac

# Build TEST-ONLY embedding cache with chr-split, NO alt alleles
# --test-only: skip train/val (only rebuild test embeddings)
# --chr-split: use chromosome-based splits (chr7+13 test)
# NO --include-alt-alleles: ref-only test set (~31K sequences)
echo "=== Building test-only embedding cache ==="
uv run --no-sync python "${BUILD_SCRIPT}" \
    --data-path "data/k562" \
    --cache-dir "${CACHE_DIR}" \
    --test-only \
    --chr-split \
    --cell-line "${CELL}" \
    ${EXTRA_FLAGS}

echo ""
echo "=== Cache files ==="
ls -lh "${CACHE_DIR}"/test_* 2>/dev/null || echo "(no test files found)"

echo "=== task=${T} DONE — $(date) ==="
