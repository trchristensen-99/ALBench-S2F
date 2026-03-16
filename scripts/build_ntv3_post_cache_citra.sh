#!/bin/bash
# Build NTv3 post-trained (Borzoi-finetuned) embedding cache on Citra.
# Parallelizes embedding extraction across multiple GPUs, then trains
# S1 heads on the cached embeddings.
#
# Usage (from repo root):
#   bash scripts/build_ntv3_post_cache_citra.sh
#
# Prerequisites:
#   - NTv3 model weights (auto-downloaded from HuggingFace on first run)
#   - K562 data at data/k562/
#   - external/nucleotide-transformer/ installed (pip install -e)
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

CACHE_DIR="outputs/ntv3_post_k562_cached/embedding_cache"
BATCH_SIZE=16

echo "=== NTv3 Post-trained Embedding Cache Build (Citra, multi-GPU) ==="
echo "Cache dir: $CACHE_DIR"
echo "Date: $(date)"
mkdir -p logs "$CACHE_DIR"

N_GPUS=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 0)
echo "Available GPUs: $N_GPUS"

if [ "$N_GPUS" -eq 0 ]; then
    echo "ERROR: No GPUs detected."
    exit 1
fi

# ── Phase 1: Build embedding caches (parallel across GPUs) ───────────
# GPU 0: train (largest, ~294K seqs — bottleneck)
# GPU 1: val (~37K seqs)
# GPU 2: test sets (~135K seqs total)
# If fewer GPUs, run sequentially.

echo ""
echo "=== Phase 1: Embedding cache (train + val + test) ==="

CUDA_VISIBLE_DEVICES=0 python scripts/build_ntv3_embedding_cache.py \
    --model-variant post \
    --cache-dir "$CACHE_DIR" \
    --batch-size "$BATCH_SIZE" \
    --splits train \
    2>&1 | tee logs/ntv3_post_cache_train.log &
PID_TRAIN=$!
echo "  Train started on GPU 0 (PID $PID_TRAIN)"

if [ "$N_GPUS" -ge 2 ]; then
    CUDA_VISIBLE_DEVICES=1 python scripts/build_ntv3_embedding_cache.py \
        --model-variant post \
        --cache-dir "$CACHE_DIR" \
        --batch-size "$BATCH_SIZE" \
        --splits val \
        2>&1 | tee logs/ntv3_post_cache_val.log &
    PID_VAL=$!
    echo "  Val started on GPU 1 (PID $PID_VAL)"
else
    PID_VAL=""
fi

if [ "$N_GPUS" -ge 3 ]; then
    CUDA_VISIBLE_DEVICES=2 python scripts/build_ntv3_embedding_cache.py \
        --model-variant post \
        --cache-dir "$CACHE_DIR" \
        --batch-size "$BATCH_SIZE" \
        --splits val \
        --include-test \
        2>&1 | tee logs/ntv3_post_cache_test.log &
    PID_TEST=$!
    echo "  Test sets started on GPU 2 (PID $PID_TEST)"
else
    PID_TEST=""
fi

# Wait for all embedding jobs
echo ""
echo "Waiting for embedding cache builds..."
wait $PID_TRAIN
echo "  Train done."
[ -n "${PID_VAL:-}" ] && wait $PID_VAL && echo "  Val done."
[ -n "${PID_TEST:-}" ] && wait $PID_TEST && echo "  Test done."

# If test wasn't run in parallel, run it now
if [ -z "${PID_TEST:-}" ]; then
    echo "  Running val + test sequentially..."
    CUDA_VISIBLE_DEVICES=0 python scripts/build_ntv3_embedding_cache.py \
        --model-variant post \
        --cache-dir "$CACHE_DIR" \
        --batch-size "$BATCH_SIZE" \
        --splits val \
        --include-test \
        2>&1 | tee logs/ntv3_post_cache_test.log
fi

echo ""
echo "=== Embedding cache complete ==="
ls -lh "$CACHE_DIR"/*.npy 2>/dev/null

# ── Phase 2: Train S1 heads (3 seeds) ────────────────────────────────
echo ""
echo "=== Phase 2: Train S1 heads (3 seeds) ==="

# NTv3 embed_dim = 1536
SEEDS=(42 123 456)
PIDS=()
for i in "${!SEEDS[@]}"; do
    SEED=${SEEDS[$i]}
    GPU=$((i % N_GPUS))
    OUT_DIR="outputs/ntv3_post_k562_3seeds/seed_${SEED}"

    CUDA_VISIBLE_DEVICES=$GPU python experiments/train_foundation_cached.py \
        ++model_name=ntv3_post \
        ++cache_dir="$CACHE_DIR" \
        ++embed_dim=1536 \
        ++output_dir="$OUT_DIR" \
        ++seed=$SEED \
        ++epochs=100 \
        ++early_stop_patience=10 \
        ++hidden_dim=512 \
        ++dropout=0.1 \
        ++lr=0.001 \
        ++batch_size=512 \
        2>&1 | tee "logs/ntv3_post_s1_seed${SEED}.log" &
    PIDS+=($!)
    echo "  Seed $SEED on GPU $GPU (PID ${PIDS[-1]})"
done

echo "Waiting for S1 training..."
for pid in "${PIDS[@]}"; do
    wait $pid
done
echo "  All seeds done."

echo ""
echo "=== Results ==="
for d in outputs/ntv3_post_k562_3seeds/seed_*/; do
    if [ -f "$d/result.json" ]; then
        python -c "
import json
d = json.load(open('${d}result.json'))
tm = d.get('test_metrics', d)
id_r = tm.get('in_distribution', tm.get('in_dist', {})).get('pearson_r', '?')
ood_r = tm.get('ood', {}).get('pearson_r', '?')
print(f'  $(basename $d): in_dist={id_r}, ood={ood_r}')
" 2>/dev/null
    fi
done

echo ""
echo "=== All done: $(date) ==="
