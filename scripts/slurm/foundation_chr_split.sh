#!/bin/bash
# Foundation models on chromosome-based splits.
# Step 1: Build chr-split embedding caches (uses chr7+13 test, chr19+21+X val)
# Step 2: Train S1 heads on the chr-split caches (3 seeds each)
#
# Array:
#   0: Enformer K562 chr-split (cache + 3 seeds)
#   1: Borzoi K562 chr-split (cache + 3 seeds)
#   2: NTv3 K562 chr-split (cache + 3 seeds)
#   3: Enformer HepG2 chr-split (cache + 3 seeds)
#   4: Borzoi HepG2 chr-split (cache + 3 seeds)
#   5: NTv3 HepG2 chr-split (cache + 3 seeds)
#   6: Enformer SKNSH chr-split (cache + 3 seeds)
#   7: Borzoi SKNSH chr-split (cache + 3 seeds)
#   8: NTv3 SKNSH chr-split (cache + 3 seeds)
#
#SBATCH --job-name=fdn_chr
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --time=48:00:00
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
echo "=== foundation chr-split task=${T} node=${SLURMD_NODENAME} date=$(date) ==="

MODELS=("enformer" "borzoi" "ntv3_post" "enformer" "borzoi" "ntv3_post" "enformer" "borzoi" "ntv3_post")
CELLS=("k562" "k562" "k562" "hepg2" "hepg2" "hepg2" "sknsh" "sknsh" "sknsh")

MODEL="${MODELS[$T]}"
CELL="${CELLS[$T]}"
CACHE_DIR="outputs/chr_split/${CELL}/${MODEL}_cached/embedding_cache"
OUT_DIR="outputs/chr_split/${CELL}/${MODEL}_s1"

echo "Model: ${MODEL}, Cell: ${CELL}"
echo "Cache: ${CACHE_DIR}"
echo "Output: ${OUT_DIR}"

# Determine embed dim and build script
case ${MODEL} in
    enformer)
        EMBED_DIM=3072
        BUILD_SCRIPT="scripts/build_enformer_embedding_cache.py"
        ;;
    borzoi)
        EMBED_DIM=1536
        BUILD_SCRIPT="scripts/build_borzoi_embedding_cache.py"
        ;;
    ntv3_post)
        EMBED_DIM=1536
        BUILD_SCRIPT="scripts/build_ntv3_embedding_cache.py"
        ;;
esac

# Step 1: Build embedding cache with chr-split
echo "=== Building embedding cache ==="
uv run --no-sync python "${BUILD_SCRIPT}" \
    --data-path "data/k562" \
    --cache-dir "${CACHE_DIR}" \
    --splits train val \
    --include-test \
    --chr-split \
    --cell-line "${CELL}"

# Step 2: Train S1 heads (3 seeds)
echo "=== Training S1 heads ==="
for SEED in 42 123 456; do
    echo "--- Seed ${SEED} ---"
    uv run --no-sync python experiments/train_foundation_cached.py \
        ++model_name="${MODEL}" \
        ++cell_line="${CELL}" \
        ++cache_dir="${CACHE_DIR}" \
        ++embed_dim="${EMBED_DIM}" \
        ++output_dir="${OUT_DIR}/seed_${SEED}" \
        ++seed="${SEED}" \
        ++chr_split=True \
        ++lr=0.0005 ++weight_decay=1e-6 ++dropout=0.1 \
        ++epochs=50 ++early_stop_patience=7
done

echo "=== Done: $(date) ==="
