#!/bin/bash
# Multi-task training + high-activity duplication experiments.
#
# Tests two boda2/Malinois paper techniques:
#   1. Multi-task: train single head for all 3 cell types jointly
#   2. High-activity duplication: duplicate training sequences with label >= 0.5
#
# Array:
#   0: Enformer S1 multi-task (no duplication)
#   1: Enformer S1 multi-task + duplication_cutoff=0.5
#   2: Enformer S1 single-task + duplication_cutoff=0.5 (K562 only, for comparison)
#   3: Malinois multi-task (K562 only, duplication=0.5) — requires code for Malinois multi-task
#   4: DREAM-RNN + duplication_cutoff=0.5 (K562 chr_split, ref+alt)
#
# Submit:
#   sbatch --array=0-2 --qos=default --time=12:00:00 scripts/slurm/multitask_and_duplication.sh
#   sbatch --array=3-4 --qos=slow_nice scripts/slurm/multitask_and_duplication.sh
#
#SBATCH --job-name=mt_dup
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
echo "=== mt_dup task=${T} node=${SLURMD_NODENAME} date=$(date) ==="

# Setup data symlinks
for CELL in hepg2 sknsh; do
    mkdir -p "data/${CELL}"
    ln -sf "$(pwd)/data/k562/DATA-Table_S2__MPRA_dataset.txt" "data/${CELL}/DATA-Table_S2__MPRA_dataset.txt" 2>/dev/null || true
    ln -sf "$(pwd)/data/k562/hashfrag_splits" "data/${CELL}/hashfrag_splits" 2>/dev/null || true
    ln -sf "$(pwd)/data/k562/test_sets" "data/${CELL}/test_sets" 2>/dev/null || true
done

# Use existing chr_split embedding cache (ref+alt)
CACHE_DIR="outputs/chr_split/k562/enformer_cached_v2/embedding_cache"

# If v2 cache not ready yet, fall back to rebuilding
if [ ! -f "${CACHE_DIR}/train_canonical.npy" ]; then
    echo "Cache not found at ${CACHE_DIR}, building..."
    CACHE_DIR="outputs/multitask/enformer_cached/embedding_cache"
    uv run --no-sync python scripts/build_enformer_embedding_cache.py \
        --data-path "data/k562" \
        --cache-dir "${CACHE_DIR}" \
        --splits train val \
        --include-test \
        --chr-split \
        --cell-line k562 \
        --include-alt-alleles
fi

case ${T} in

0)  # Enformer S1 multi-task (no duplication)
    echo "Enformer S1 multi-task (3 cell types, no duplication)"
    for SEED in 42 123 456; do
        echo "--- Seed ${SEED} ---"
        uv run --no-sync python experiments/train_foundation_cached_multitask.py \
            ++model_name=enformer \
            ++cache_dir="${CACHE_DIR}" \
            ++embed_dim=3072 \
            ++output_dir="outputs/multitask/enformer_s1_mt/seed_${SEED}" \
            ++seed="${SEED}" \
            ++chr_split=True \
            ++include_alt_alleles=True \
            ++lr=0.0005 ++weight_decay=1e-6 ++dropout=0.1 \
            ++epochs=50 ++early_stop_patience=7
    done
    ;;

1)  # Enformer S1 multi-task + duplication
    echo "Enformer S1 multi-task + duplication_cutoff=0.5"
    for SEED in 42 123 456; do
        echo "--- Seed ${SEED} ---"
        uv run --no-sync python experiments/train_foundation_cached_multitask.py \
            ++model_name=enformer \
            ++cache_dir="${CACHE_DIR}" \
            ++embed_dim=3072 \
            ++output_dir="outputs/multitask/enformer_s1_mt_dup/seed_${SEED}" \
            ++seed="${SEED}" \
            ++chr_split=True \
            ++include_alt_alleles=True \
            ++duplication_cutoff=0.5 \
            ++lr=0.0005 ++weight_decay=1e-6 ++dropout=0.1 \
            ++epochs=50 ++early_stop_patience=7
    done
    ;;

2)  # Enformer S1 single-task K562 + duplication (ablation)
    echo "Enformer S1 single-task K562 + duplication_cutoff=0.5"
    for SEED in 42 123 456; do
        echo "--- Seed ${SEED} ---"
        uv run --no-sync python experiments/train_foundation_cached.py \
            ++model_name=enformer \
            ++cell_line=k562 \
            ++cache_dir="${CACHE_DIR}" \
            ++embed_dim=3072 \
            ++output_dir="outputs/multitask/enformer_s1_dup_only/seed_${SEED}" \
            ++seed="${SEED}" \
            ++chr_split=True \
            ++include_alt_alleles=True \
            ++duplication_cutoff=0.5 \
            ++lr=0.0005 ++weight_decay=1e-6 ++dropout=0.1 \
            ++epochs=50 ++early_stop_patience=7
    done
    ;;

3)  # Malinois K562 + duplication
    echo "Malinois K562 + duplication_cutoff=0.5"
    for SEED in 0 1 2; do
        echo "--- seed ${SEED} ---"
        uv run --no-sync python experiments/train_malinois_k562.py \
            ++data_path="data/k562" \
            ++output_dir="outputs/multitask/malinois_dup/seed_${SEED}" \
            ++seed="${SEED}" \
            ++cell_line="k562" \
            ++chr_split=True \
            ++include_alt_alleles=True \
            ++duplication_cutoff=0.5
    done
    ;;

4)  # DREAM-RNN K562 + duplication
    echo "DREAM-RNN K562 + duplication_cutoff=0.5"
    uv run --no-sync python experiments/exp1_1_scaling.py \
        --task k562 --student dream_rnn \
        --oracle ground_truth --reservoir genomic --chr-split \
        --include-alt-alleles \
        --n-replicates 3 --seed 42 \
        --output-dir "outputs/multitask/dream_rnn_dup" \
        --training-sizes 700000 --epochs 80 --ensemble-size 1 \
        --early-stop-patience 10 \
        --duplication-cutoff 0.5
    ;;

*)  echo "ERROR: unknown task ${T}"; exit 1 ;;
esac

echo "=== task=${T} DONE — $(date) ==="
