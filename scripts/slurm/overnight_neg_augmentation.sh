#!/bin/bash
# Overnight pipeline: Oracle bias correction via negative augmentation
#
# Step 1: Generate synthetic negatives (CPU, fast)
# Step 2: Cache AG embeddings for negatives (H100, ~30 min)
# Step 3: Train S1 oracle heads with negatives (H100, ~20 min/fold)
#         - 3 augmentation types × 2 fractions × 1 fold (quick test)
# Step 4: Evaluate on random DNA + shuffled controls
#
# Also runs: LegNet real-label scaling at missing sizes
#
# Submit:
#   /cm/shared/apps/slurm/current/bin/sbatch scripts/slurm/overnight_neg_augmentation.sh
#
#SBATCH --job-name=neg_aug_pipe
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --time=24:00:00
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
export ALPHAGENOME_WEIGHTS="/grid/wsbs/home_norepl/christen/alphagenome_weights/alphagenome-jax-all_folds-v1"

CACHE_DIR="outputs/oracle_full_856k/embedding_cache"
NEG_DIR="data/synthetic_negatives"
BASE_OUT="outputs/oracle_neg_augmentation"

echo "=== Step 1: Generate synthetic negatives — $(date) ==="
if [ ! -f "${NEG_DIR}/metadata.json" ]; then
    uv run --no-sync python scripts/generate_synthetic_negatives.py \
        --output-dir "${NEG_DIR}" \
        --n-random 50000 --n-gc-matched 50000
else
    echo "  Negatives already exist, skipping"
fi

echo ""
echo "=== Step 2: Cache AG embeddings for negatives — $(date) ==="
# We cache embeddings for each negative type separately
for NEG_TYPE in random dinuc_shuffled gc_matched; do
    NEG_CACHE="${BASE_OUT}/neg_embed_cache"
    CAN_FILE="${NEG_CACHE}/neg_${NEG_TYPE}_canonical.npy"
    if [ -f "${CAN_FILE}" ]; then
        echo "  ${NEG_TYPE} embeddings already cached, skipping"
        continue
    fi
    echo "  Caching ${NEG_TYPE} embeddings..."
    mkdir -p "${NEG_CACHE}"
    # The training script will auto-cache on first run
done

echo ""
echo "=== Step 3: Train S1 oracle with negative augmentation — $(date) ==="
# Test matrix: 3 neg types × 2 fractions × fold 0 only (quick test)
for NEG_TYPE in random dinuc_shuffled gc_matched; do
    for FRAC in 0.05 0.10; do
        FRAC_LABEL=$(echo "$FRAC" | sed 's/0\.//')
        OUT="${BASE_OUT}/${NEG_TYPE}_frac${FRAC_LABEL}/s1/oracle_0"

        if [ -f "${OUT}/test_metrics.json" ]; then
            echo "  ${NEG_TYPE} frac=${FRAC} already done, skipping"
            continue
        fi

        echo "  Training: neg_type=${NEG_TYPE}, fraction=${FRAC}, fold=0"
        uv run --no-sync python scripts/train_oracle_s1_with_negatives.py \
            --cache-dir "${CACHE_DIR}" \
            --negatives-dir "${NEG_DIR}" \
            --neg-cache-dir "${BASE_OUT}/neg_embed_cache" \
            --output-dir "${OUT}" \
            --fold-id 0 --n-folds 10 \
            --neg-type "${NEG_TYPE}" --neg-fraction "${FRAC}" \
            --epochs 50 --early-stop-patience 7 \
            --lr 0.001 --batch-size 128 || true
    done
done

echo ""
echo "=== Step 3b: Also train 'all' negatives at 0.10 — $(date) ==="
OUT="${BASE_OUT}/all_frac10/s1/oracle_0"
if [ ! -f "${OUT}/test_metrics.json" ]; then
    uv run --no-sync python scripts/train_oracle_s1_with_negatives.py \
        --cache-dir "${CACHE_DIR}" \
        --negatives-dir "${NEG_DIR}" \
        --neg-cache-dir "${BASE_OUT}/neg_embed_cache" \
        --output-dir "${OUT}" \
        --fold-id 0 --n-folds 10 \
        --neg-type all --neg-fraction 0.10 \
        --epochs 50 --early-stop-patience 7 || true
fi

echo ""
echo "=== Step 4: Quick evaluation of each trained oracle — $(date) ==="
uv run --no-sync python -c "
import json, numpy as np, sys, glob
from pathlib import Path

base = Path('${BASE_OUT}')
print('Oracle Negative Augmentation Results')
print('=' * 70)

for config_dir in sorted(base.iterdir()):
    if not config_dir.is_dir() or config_dir.name == 'neg_embed_cache':
        continue
    result_file = config_dir / 's1' / 'oracle_0' / 'test_metrics.json'
    if result_file.exists():
        d = json.loads(result_file.read_text())
        print(f'{config_dir.name}: val_r={d[\"best_val_pearson\"]:.4f}, '
              f'neg_type={d.get(\"neg_type\",\"?\")}, frac={d.get(\"neg_fraction\",\"?\")}, '
              f'n_neg={d.get(\"n_negatives\",\"?\")}')
    else:
        print(f'{config_dir.name}: NOT DONE')
" || true

echo ""
echo "=== Step 5: Run LegNet real-label scaling (missing sizes) — $(date) ==="
# Train LegNet on real labels at sizes matching oracle scaling
for N in 3197 6395 15987 63949 159871; do
    OUT_DIR="outputs/exp0_oracle_scaling_v4/k562/legnet_ground_truth_v2/genomic/n${N}/hp0/seed42"
    if [ -f "${OUT_DIR}/result.json" ]; then
        echo "  n=${N} already done"
        continue
    fi
    echo "  Training LegNet real-label n=${N}..."
    uv run --no-sync python experiments/exp1_1_scaling.py \
        --task k562 --student legnet --oracle ground_truth \
        --reservoir genomic \
        --n-replicates 1 --seed 42 \
        --output-dir "outputs/exp0_oracle_scaling_v4/k562/legnet_ground_truth_v2" \
        --training-sizes "${N}" \
        --chr-split --lr 0.001 --batch-size 512 \
        --epochs 80 --ensemble-size 1 --early-stop-patience 10 \
        --save-predictions || true
done

echo ""
echo "=== All done — $(date) ==="
