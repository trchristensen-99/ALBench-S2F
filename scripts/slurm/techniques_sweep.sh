#!/bin/bash
# Comprehensive sweep of boda2/paper techniques across all model types.
# All use chr_split + ref+alt (matching paper). K562 only for efficiency.
# 2 seeds where noted to save time.
#
# Array:
# --- AG (fast: ~20-35 min each) ---
#   0: AG S1 multi-task (3 cell types, 1 seed)
#   1: AG S1 multi-task + duplication (1 seed)
#   2: AG S1 single-task + duplication (1 seed, ablation)
#   3: AG S2 from multi-task S1 (warm start, 1 seed)
#
# --- Enformer (medium: ~1-2h each, needs cache) ---
#   4: Enformer S1 multi-task (3 seeds)
#   5: Enformer S1 multi-task + duplication (3 seeds)
#   6: Enformer S1 single-task + duplication (3 seeds, ablation)
#
# --- Malinois (medium: ~30 min/seed) ---
#   7: Malinois + duplication + cosine LR (2 seeds)
#   8: Malinois + RC interleave + duplication (2 seeds)
#   9: Malinois + RC interleave + cosine LR + duplication (2 seeds, full boda2)
#
# --- DREAM-RNN (slow: ~3h/seed) ---
#  10: DREAM-RNN + duplication (2 seeds)
#
# Submit (spread across QoS for max parallelism):
#   sbatch --array=0-3,7-9 --qos=default --time=12:00:00 scripts/slurm/techniques_sweep.sh
#   sbatch --array=4-6,10 --qos=slow_nice scripts/slurm/techniques_sweep.sh
#
#SBATCH --job-name=tech_sweep
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
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_command_buffer="

T=$SLURM_ARRAY_TASK_ID
echo "=== techniques_sweep task=${T} node=${SLURMD_NODENAME} date=$(date) ==="

# Setup data symlinks
for CELL in hepg2 sknsh; do
    mkdir -p "data/${CELL}"
    ln -sf "$(pwd)/data/k562/DATA-Table_S2__MPRA_dataset.txt" "data/${CELL}/DATA-Table_S2__MPRA_dataset.txt" 2>/dev/null || true
    ln -sf "$(pwd)/data/k562/hashfrag_splits" "data/${CELL}/hashfrag_splits" 2>/dev/null || true
    ln -sf "$(pwd)/data/k562/test_sets" "data/${CELL}/test_sets" 2>/dev/null || true
done

OUT_BASE="outputs/techniques_sweep"

# Enformer cache (shared across Enformer experiments)
ENF_CACHE="outputs/chr_split/k562/enformer_cached_v2/embedding_cache"

case ${T} in

# ── AG S1 multi-task ──
0)  echo "AG S1 multi-task (3 cell types)"
    uv run --no-sync python experiments/exp1_1_scaling.py \
        --task k562 --student alphagenome_k562_s1 --multitask \
        --oracle ground_truth --reservoir genomic --chr-split \
        --include-alt-alleles \
        --n-replicates 1 --no-hp-sweep --seed 42 \
        --output-dir "${OUT_BASE}/ag_s1_multitask" \
        --training-sizes 658000 --epochs 50 --early-stop-patience 7
    ;;

1)  echo "AG S1 multi-task + duplication"
    uv run --no-sync python experiments/exp1_1_scaling.py \
        --task k562 --student alphagenome_k562_s1 --multitask \
        --oracle ground_truth --reservoir genomic --chr-split \
        --include-alt-alleles --duplication-cutoff 0.5 \
        --n-replicates 1 --no-hp-sweep --seed 42 \
        --output-dir "${OUT_BASE}/ag_s1_multitask_dup" \
        --training-sizes 658000 --epochs 50 --early-stop-patience 7
    ;;

2)  echo "AG S1 single-task + duplication (ablation)"
    uv run --no-sync python experiments/exp1_1_scaling.py \
        --task k562 --student alphagenome_k562_s1 \
        --oracle ground_truth --reservoir genomic --chr-split \
        --include-alt-alleles --duplication-cutoff 0.5 \
        --n-replicates 1 --no-hp-sweep --seed 42 \
        --output-dir "${OUT_BASE}/ag_s1_dup" \
        --training-sizes 658000 --epochs 50 --early-stop-patience 7
    ;;

3)  echo "AG S2 from multi-task S1 (warm start)"
    # Run S1 multi-task first if no checkpoint
    S1_CKPT="${OUT_BASE}/ag_s1_multitask/genomic/n700000/hp0/seed42/best_model/checkpoint"
    if [ ! -d "${S1_CKPT}" ]; then
        echo "  Running AG S1 multi-task first..."
        uv run --no-sync python experiments/exp1_1_scaling.py \
            --task k562 --student alphagenome_k562_s1 --multitask \
            --oracle ground_truth --reservoir genomic --chr-split \
            --include-alt-alleles \
            --n-replicates 1 --no-hp-sweep --seed 42 \
            --output-dir "${OUT_BASE}/ag_s1_multitask" \
            --training-sizes 658000 --epochs 50 --early-stop-patience 7
    fi
    echo "  Running AG S2 with warm start from multi-task S1"
    uv run --no-sync python experiments/exp1_1_scaling.py \
        --task k562 --student alphagenome_k562_s2 \
        --oracle ground_truth --reservoir genomic --chr-split \
        --include-alt-alleles \
        --s1-checkpoint "${OUT_BASE}/ag_s1_multitask/genomic/n700000/hp0/seed42" \
        --n-replicates 1 --no-hp-sweep --seed 42 \
        --output-dir "${OUT_BASE}/ag_s2_from_mt" \
        --training-sizes 658000 --epochs 50 --early-stop-patience 7
    ;;

# ── Enformer S1 multi-task ──
4)  echo "Enformer S1 multi-task (3 cell types)"
    for SEED in 42 123; do
        echo "--- Seed ${SEED} ---"
        uv run --no-sync python experiments/train_foundation_cached_multitask.py \
            ++model_name=enformer \
            ++cache_dir="${ENF_CACHE}" \
            ++embed_dim=3072 \
            ++output_dir="${OUT_BASE}/enformer_s1_multitask/seed_${SEED}" \
            ++seed="${SEED}" \
            ++chr_split=True \
            ++include_alt_alleles=True \
            ++lr=0.0005 ++weight_decay=1e-6 ++dropout=0.1 \
            ++epochs=50 ++early_stop_patience=7
    done
    ;;

5)  echo "Enformer S1 multi-task + duplication"
    for SEED in 42 123; do
        echo "--- Seed ${SEED} ---"
        uv run --no-sync python experiments/train_foundation_cached_multitask.py \
            ++model_name=enformer \
            ++cache_dir="${ENF_CACHE}" \
            ++embed_dim=3072 \
            ++output_dir="${OUT_BASE}/enformer_s1_multitask_dup/seed_${SEED}" \
            ++seed="${SEED}" \
            ++chr_split=True \
            ++include_alt_alleles=True \
            ++duplication_cutoff=0.5 \
            ++lr=0.0005 ++weight_decay=1e-6 ++dropout=0.1 \
            ++epochs=50 ++early_stop_patience=7
    done
    ;;

6)  echo "Enformer S1 single-task + duplication (ablation)"
    for SEED in 42 123; do
        echo "--- Seed ${SEED} ---"
        uv run --no-sync python experiments/train_foundation_cached.py \
            ++model_name=enformer \
            ++cell_line=k562 \
            ++cache_dir="${ENF_CACHE}" \
            ++embed_dim=3072 \
            ++output_dir="${OUT_BASE}/enformer_s1_dup/seed_${SEED}" \
            ++seed="${SEED}" \
            ++chr_split=True \
            ++include_alt_alleles=True \
            ++duplication_cutoff=0.5 \
            ++lr=0.0005 ++weight_decay=1e-6 ++dropout=0.1 \
            ++epochs=50 ++early_stop_patience=7
    done
    ;;

# ── Malinois technique combos ──
7)  echo "Malinois + duplication + cosine LR"
    for SEED in 0 1; do
        echo "--- seed ${SEED} ---"
        uv run --no-sync python experiments/train_malinois_k562.py \
            ++data_path="data/k562" \
            ++output_dir="${OUT_BASE}/malinois_dup_cosine/seed_${SEED}" \
            ++seed="${SEED}" \
            ++cell_line=k562 \
            ++chr_split=True \
            ++include_alt_alleles=True \
            ++duplication_cutoff=0.5 \
            ++lr_schedule=cosine
    done
    ;;

8)  echo "Malinois + RC interleave + duplication"
    for SEED in 0 1; do
        echo "--- seed ${SEED} ---"
        uv run --no-sync python experiments/train_malinois_k562.py \
            ++data_path="data/k562" \
            ++output_dir="${OUT_BASE}/malinois_interleave_dup/seed_${SEED}" \
            ++seed="${SEED}" \
            ++cell_line=k562 \
            ++chr_split=True \
            ++include_alt_alleles=True \
            ++duplication_cutoff=0.5 \
            ++rc_mode=interleave
    done
    ;;

9)  echo "Malinois full boda2 (RC interleave + cosine LR + duplication)"
    for SEED in 0 1; do
        echo "--- seed ${SEED} ---"
        uv run --no-sync python experiments/train_malinois_k562.py \
            ++data_path="data/k562" \
            ++output_dir="${OUT_BASE}/malinois_full_boda2/seed_${SEED}" \
            ++seed="${SEED}" \
            ++cell_line=k562 \
            ++chr_split=True \
            ++include_alt_alleles=True \
            ++duplication_cutoff=0.5 \
            ++rc_mode=interleave \
            ++lr_schedule=cosine
    done
    ;;

# ── DREAM-RNN + duplication ──
10) echo "DREAM-RNN + duplication"
    uv run --no-sync python experiments/exp1_1_scaling.py \
        --task k562 --student dream_rnn \
        --oracle ground_truth --reservoir genomic --chr-split \
        --include-alt-alleles --duplication-cutoff 0.5 \
        --n-replicates 2 --seed 42 \
        --output-dir "${OUT_BASE}/dream_rnn_dup" \
        --training-sizes 700000 --epochs 80 --ensemble-size 1 \
        --early-stop-patience 10
    ;;

*)  echo "ERROR: unknown task ${T}"; exit 1 ;;
esac

echo "=== task=${T} DONE — $(date) ==="
