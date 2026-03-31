#!/bin/bash
# Borzoi S1 embedding extraction sweep.
# Instead of varying S2 fine-tuning (which consistently fails),
# test different embedding extraction strategies with S1 (frozen encoder + head).
#
# Key insight: PRE-TRANSFORMER center bins have 6% relative L2 distance
# (vs <0.5% post-transformer), because conv layers are LOCAL while
# attention is GLOBAL. The insert maps to ~5 pre-transformer bins.
#
# Array:
#   0: Pre-transformer center 5 bins (analog of AG's 5 tokens)
#   1: Pre-transformer center 10 bins
#   2: Pre-transformer center 20 bins
#   3: x_unet0 center 20 bins (1280-dim, before unet1)
#   4: x_unet1 center 10 bins (1536-dim, before max_pool)
#   5: Post-transformer center 5 bins (for comparison)
#   6: Post-transformer center 20 bins (for comparison)
#   7: Full pipeline all-bins mean-pool (current S1 baseline)
#
# All use V100 (frozen encoder, fast)
#
#SBATCH --job-name=borz_emb
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=default
#SBATCH --time=4:00:00
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
echo "=== Borzoi embedding sweep task=${T} node=${SLURMD_NODENAME} date=$(date) ==="

# Each task: build cache with specific extraction mode, then train 3-seed heads
CONFIGS=(
    "pre_transformer_c5"
    "pre_transformer_c10"
    "pre_transformer_c20"
    "unet0_c20"
    "unet1_c10"
    "post_transformer_c5"
    "post_transformer_c20"
    "full_pipeline_allbins"
)
CONFIG="${CONFIGS[$T]}"
CACHE_DIR="outputs/borzoi_emb_sweep/${CONFIG}/embedding_cache"
OUT_DIR="outputs/borzoi_emb_sweep/${CONFIG}"

echo "Config: ${CONFIG}"
echo "Cache: ${CACHE_DIR}"

# Step 1: Build cache with this extraction mode
uv run --no-sync python scripts/build_borzoi_embedding_cache.py \
    --data-path data/k562 \
    --cache-dir "${CACHE_DIR}" \
    --include-test \
    --extraction-mode "${CONFIG}"

# Step 2: Train 3-seed heads
# Determine embed_dim based on extraction mode
case ${CONFIG} in
    unet0_*)
        EMBED_DIM=1280
        ;;
    *)
        EMBED_DIM=1536
        ;;
esac

for SEED in 42 123 456; do
    echo "--- Seed ${SEED} ---"
    uv run --no-sync python experiments/train_foundation_cached.py \
        ++model_name=borzoi \
        ++cell_line=k562 \
        ++cache_dir="${CACHE_DIR}" \
        ++embed_dim="${EMBED_DIM}" \
        ++output_dir="${OUT_DIR}/seed_${SEED}" \
        ++seed=${SEED} \
        ++lr=0.0005 ++weight_decay=1e-6 ++dropout=0.1 \
        ++epochs=100 ++early_stop_patience=10
done

echo "=== Done: $(date) ==="
