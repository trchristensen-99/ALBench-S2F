#!/bin/bash
# Complete chr_split experiment suite — fills all gaps vs HashFrag.
#
# PHASE 1: Re-run foundation S1 with corrected test evaluation (chr7+13 in-dist,
#           filtered SNV, K562-only OOD). Previous runs used hashfrag test sets.
# PHASE 2: Enformer S2 chr_split (missing entirely).
#
# Array:
#   0-8:   Foundation S1 re-eval (cache rebuild + 3-seed head training)
#          0=Enformer K562, 1=Borzoi K562, 2=NTv3 K562,
#          3=Enformer HepG2, 4=Borzoi HepG2, 5=NTv3 HepG2,
#          6=Enformer SknSh, 7=Borzoi SknSh, 8=NTv3 SknSh
#   9-11:  Enformer S2 chr_split (3 cells × 1 config)
#          9=K562, 10=HepG2, 11=SknSh
#
# Submit:
#   /cm/shared/apps/slurm/current/bin/sbatch --array=0-11 scripts/slurm/chr_split_complete.sh
#
# Use all 3 QoS tiers for max parallelism:
#   sbatch --array=0-1 --qos=fast scripts/slurm/chr_split_complete.sh
#   sbatch --array=2-5 --qos=default scripts/slurm/chr_split_complete.sh
#   sbatch --array=6-11 --qos=slow_nice scripts/slurm/chr_split_complete.sh
#
#SBATCH --job-name=chr_complete
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
echo "=== chr_split_complete task=${T} node=${SLURMD_NODENAME} date=$(date) ==="

# Setup data symlinks for all cells
for CELL in hepg2 sknsh; do
    mkdir -p "data/${CELL}"
    ln -sf "$(pwd)/data/k562/DATA-Table_S2__MPRA_dataset.txt" "data/${CELL}/DATA-Table_S2__MPRA_dataset.txt" 2>/dev/null || true
    ln -sf "$(pwd)/data/k562/hashfrag_splits" "data/${CELL}/hashfrag_splits" 2>/dev/null || true
    ln -sf "$(pwd)/data/k562/test_sets" "data/${CELL}/test_sets" 2>/dev/null || true
done

# ── PHASE 1: Foundation S1 re-run with corrected chr_split test eval ──
if [ $T -le 8 ]; then
    MODELS=("enformer" "borzoi" "ntv3_post" "enformer" "borzoi" "ntv3_post" "enformer" "borzoi" "ntv3_post")
    CELLS=("k562" "k562" "k562" "hepg2" "hepg2" "hepg2" "sknsh" "sknsh" "sknsh")

    MODEL="${MODELS[$T]}"
    CELL="${CELLS[$T]}"
    # Use v2 cache dir to avoid overwriting existing results
    CACHE_DIR="outputs/chr_split/${CELL}/${MODEL}_cached_v2/embedding_cache"
    OUT_DIR="outputs/chr_split/${CELL}/${MODEL}_s1_v2"

    echo "Phase 1: Foundation S1 — Model=${MODEL}, Cell=${CELL}"
    echo "Cache: ${CACHE_DIR}, Output: ${OUT_DIR}"

    case ${MODEL} in
        enformer) EMBED_DIM=3072; BUILD_SCRIPT="scripts/build_enformer_embedding_cache.py" ;;
        borzoi)   EMBED_DIM=1536; BUILD_SCRIPT="scripts/build_borzoi_embedding_cache.py" ;;
        ntv3_post) EMBED_DIM=1536; BUILD_SCRIPT="scripts/build_ntv3_embedding_cache.py" ;;
    esac

    # Step 1: Build embedding cache with chr-split + alt alleles (matching Malinois paper)
    echo "=== Building embedding cache ==="
    uv run --no-sync python "${BUILD_SCRIPT}" \
        --data-path "data/k562" \
        --cache-dir "${CACHE_DIR}" \
        --splits train val \
        --include-test \
        --chr-split \
        --cell-line "${CELL}" \
        --include-alt-alleles

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
            ++include_alt_alleles=True \
            ++lr=0.0005 ++weight_decay=1e-6 ++dropout=0.1 \
            ++epochs=50 ++early_stop_patience=7
    done

# ── PHASE 2: Enformer S2 chr_split ──
elif [ $T -le 11 ]; then
    CELLS_S2=("k562" "hepg2" "sknsh")
    IDX=$((T - 9))
    CELL="${CELLS_S2[$IDX]}"

    # Best S1 checkpoint for warm start
    S1_DIR="outputs/chr_split/${CELL}/enformer_s1_v2/seed_42"
    OUT_DIR="outputs/chr_split/${CELL}/enformer_s2"

    echo "Phase 2: Enformer S2 — Cell=${CELL}"
    echo "S1 dir: ${S1_DIR}, Output: ${OUT_DIR}"

    # If S1 v2 doesn't exist yet (running in parallel), fall back to v1
    if [ ! -d "${S1_DIR}" ]; then
        S1_DIR="outputs/chr_split/${CELL}/enformer_s1/seed_42"
        echo "S1 v2 not found, falling back to: ${S1_DIR}"
    fi

    for SEED in 42 123 456; do
        echo "--- Enformer S2 ${CELL} seed ${SEED} ---"
        uv run --no-sync python experiments/train_foundation_stage2.py \
            ++model_name=enformer \
            ++stage1_result_dir="${S1_DIR}" \
            ++output_dir="${OUT_DIR}/seed_${SEED}" \
            ++data_path="data/${CELL}" \
            ++cell_line="${CELL}" \
            ++chr_split=True \
            ++include_alt_alleles=True \
            ++seed="${SEED}" \
            ++epochs=15 \
            ++batch_size=4 \
            ++grad_accum_steps=2 \
            ++head_lr=0.001 \
            ++encoder_lr=0.0001 \
            ++weight_decay=1e-6 \
            ++hidden_dim=512 \
            ++dropout=0.1 \
            ++early_stop_patience=5 \
            ++max_train_sequences=20000 \
            ++max_val_sequences=2000 \
            ++rc_aug=True \
            ++unfreeze_mode=all \
            ++grad_clip=1.0 \
            ++amp_mode=bfloat16
    done
fi

echo "=== task=${T} DONE — $(date) ==="
