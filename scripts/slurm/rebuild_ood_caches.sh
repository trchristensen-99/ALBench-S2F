#!/bin/bash
# Rebuild OOD embedding caches for all 3 foundation models.
# The OOD TSV was regenerated (22862→22962 sequences) after caches were built.
# Only rebuilds the test_ood split (fast — ~23K sequences).
#
# Array:
#   0: Enformer K562 OOD cache
#   1: Borzoi K562 OOD cache
#   2: NTv3 K562 OOD cache
#
# Then retrains S1 heads (3 seeds each) to get corrected OOD metrics.
#
#SBATCH --job-name=fix_ood
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
echo "=== Rebuild OOD cache task=${T} node=${SLURMD_NODENAME} date=$(date) ==="

case ${T} in
0)
    echo "Enformer K562 OOD cache rebuild"
    # Back up old cache
    CACHE="outputs/enformer_k562_cached/embedding_cache"
    mv "${CACHE}/test_ood_canonical.npy" "${CACHE}/test_ood_canonical.npy.bak" 2>/dev/null || true
    mv "${CACHE}/test_ood_rc.npy" "${CACHE}/test_ood_rc.npy.bak" 2>/dev/null || true

    uv run --no-sync python scripts/build_enformer_embedding_cache.py \
        --data-path data/k562 \
        --cache-dir "${CACHE}" \
        --test-only \
        --include-test

    # Retrain heads with corrected cache
    for SEED in 42 123 456; do
        echo "--- Enformer S1 K562 seed ${SEED} ---"
        uv run --no-sync python experiments/train_foundation_cached.py \
            ++model_name=enformer \
            ++cell_line=k562 \
            ++cache_dir="${CACHE}" \
            ++embed_dim=3072 \
            ++output_dir="outputs/enformer_k562_3seeds_v2/seed_${SEED}" \
            ++seed=${SEED} \
            ++lr=0.0005 ++weight_decay=1e-6 ++dropout=0.1 \
            ++epochs=100 ++early_stop_patience=10
    done
    ;;

1)
    echo "Borzoi K562 OOD cache rebuild"
    CACHE="outputs/borzoi_k562_cached/embedding_cache"
    mv "${CACHE}/test_ood_canonical.npy" "${CACHE}/test_ood_canonical.npy.bak" 2>/dev/null || true
    mv "${CACHE}/test_ood_rc.npy" "${CACHE}/test_ood_rc.npy.bak" 2>/dev/null || true

    uv run --no-sync python scripts/build_borzoi_embedding_cache.py \
        --data-path data/k562 \
        --cache-dir "${CACHE}" \
        --test-only \
        --include-test

    for SEED in 42 123 456; do
        echo "--- Borzoi S1 K562 seed ${SEED} ---"
        uv run --no-sync python experiments/train_foundation_cached.py \
            ++model_name=borzoi \
            ++cell_line=k562 \
            ++cache_dir="${CACHE}" \
            ++embed_dim=1536 \
            ++output_dir="outputs/borzoi_k562_3seeds_v2/seed_${SEED}" \
            ++seed=${SEED} \
            ++lr=0.0005 ++weight_decay=1e-6 ++dropout=0.1 \
            ++epochs=100 ++early_stop_patience=10
    done
    ;;

2)
    echo "NTv3 K562 OOD cache rebuild"
    CACHE="outputs/ntv3_post_k562_cached/embedding_cache"
    mv "${CACHE}/test_ood_canonical.npy" "${CACHE}/test_ood_canonical.npy.bak" 2>/dev/null || true
    mv "${CACHE}/test_ood_rc.npy" "${CACHE}/test_ood_rc.npy.bak" 2>/dev/null || true

    uv run --no-sync python scripts/build_ntv3_embedding_cache.py \
        --data-path data/k562 \
        --cache-dir "${CACHE}" \
        --model-variant post \
        --test-only \
        --include-test

    for SEED in 42 123 456; do
        echo "--- NTv3 S1 K562 seed ${SEED} ---"
        uv run --no-sync python experiments/train_foundation_cached.py \
            ++model_name=ntv3_post \
            ++cell_line=k562 \
            ++cache_dir="${CACHE}" \
            ++embed_dim=1536 \
            ++output_dir="outputs/ntv3_post_k562_3seeds_v2/seed_${SEED}" \
            ++seed=${SEED} \
            ++lr=0.0005 ++weight_decay=1e-6 ++dropout=0.1 \
            ++epochs=100 ++early_stop_patience=10
    done
    ;;

*)
    echo "ERROR: unknown task ${T}"
    exit 1
    ;;
esac

echo "=== Done: $(date) ==="
