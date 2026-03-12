#!/bin/bash
# AlphaGenome fold_1 K562 bar-plot pipeline:
#   Step 1: Build fold_1 train+val embedding cache (~30-60 min)
#   Step 2: Build fold_1 test embedding cache (~15 min)
#   Step 3: Train S1 head on fold_1 cache (f=1.0, ~5 min)
#   Step 4: Train S2 fine-tuning, 3 seeds (~30 min each)
#
# Total: ~2-3h on 1x H100
#
# Submit:
#   /cm/shared/apps/slurm/current/bin/sbatch scripts/slurm/ag_fold1_k562_pipeline.sh
#
#SBATCH --job-name=ag_fold1_pipe
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpuq
#SBATCH --qos=default
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=200G
#SBATCH --time=12:00:00

set -euo pipefail

set +u
source /etc/profile.d/modules.sh
set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
source scripts/slurm/setup_hpc_deps.sh

export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_command_buffer= --xla_gpu_autotune_level=0"
export PYTHONUNBUFFERED=1

FOLD1_WEIGHTS="/grid/wsbs/home_norepl/christen/alphagenome_weights/alphagenome-jax-fold_1"
CACHE_DIR="outputs/ag_hashfrag_fold1/embedding_cache"
S1_OUTPUT="outputs/ag_fold1_s1_cached"
S2_OUTPUT="outputs/stage2_k562_fold1"

echo "=== AlphaGenome fold_1 K562 pipeline ==="
echo "Node: ${SLURMD_NODENAME}  Date: $(date)"

# ── Step 1: Build train+val embedding cache ────────────────────────────────
echo ""
echo "=== Step 1: Build fold_1 train+val embedding cache ==="
START=$(date +%s)

uv run --no-sync python scripts/analysis/build_hashfrag_embedding_cache.py \
    --weights_path "$FOLD1_WEIGHTS" \
    --cache_dir "$CACHE_DIR" \
    --splits train val \
    --dtype float16 \
    --batch_size 128

echo "Step 1 done in $(($(date +%s) - START))s"

# ── Step 2: Build test embedding cache ─────────────────────────────────────
echo ""
echo "=== Step 2: Build fold_1 test embedding cache ==="
START=$(date +%s)

uv run --no-sync python scripts/build_test_embedding_cache.py \
    --weights-path "$FOLD1_WEIGHTS" \
    --cache-dir "$CACHE_DIR" \
    --dtype float16 \
    --batch-size 256

echo "Step 2 done in $(($(date +%s) - START))s"

# ── Step 3: Train S1 head (fraction=1.0, single run) ──────────────────────
echo ""
echo "=== Step 3: Train S1 head on fold_1 cache (f=1.0) ==="
START=$(date +%s)

uv run --no-sync python experiments/exp0_k562_scaling_alphagenome_cached.py \
    ++cache_dir="$CACHE_DIR" \
    ++weights_path="$FOLD1_WEIGHTS" \
    ++output_dir="$S1_OUTPUT" \
    ++fraction=1.0 \
    ++rc_aug=true \
    ++wandb_mode=offline

echo "Step 3 done in $(($(date +%s) - START))s"

# Find the S1 run directory (latest run_*)
S1_RUN=$(find "$S1_OUTPUT/fraction_1.0000" -maxdepth 1 -name "run_*" -type d | sort | tail -1)
echo "S1 checkpoint: $S1_RUN"

# Verify S1 checkpoint exists
if [ ! -d "$S1_RUN/best_model/checkpoint" ]; then
    echo "ERROR: S1 checkpoint not found at $S1_RUN/best_model/checkpoint"
    exit 1
fi

# ── Step 4: Train S2 (3 seeds) ────────────────────────────────────────────
echo ""
echo "=== Step 4: Train S2 fine-tuning (3 seeds) ==="
START=$(date +%s)

for SEED_IDX in 0 1 2; do
    echo ""
    echo "--- S2 seed ${SEED_IDX} ---"
    uv run --no-sync python experiments/train_stage2_k562_hashfrag.py \
        --config-name stage2_k562_full_train \
        ++weights_path="$FOLD1_WEIGHTS" \
        ++stage1_dir="$S1_RUN" \
        ++output_dir="${S2_OUTPUT}/run_${SEED_IDX}" \
        ++wandb_mode=offline
done

echo ""
echo "Step 4 done in $(($(date +%s) - START))s"

# ── Summary ────────────────────────────────────────────────────────────────
echo ""
echo "=== Pipeline COMPLETE — $(date) ==="
echo ""
echo "Results:"
for RUN in "$S2_OUTPUT"/run_*/; do
    if [ -f "$RUN/test_metrics.json" ]; then
        echo "  $RUN:"
        python3 -c "
import json
m = json.load(open('${RUN}/test_metrics.json'))
for k in ['in_distribution', 'snv_abs', 'snv_delta', 'ood']:
    if k in m:
        print(f'    {k}: {m[k][\"pearson_r\"]:.4f}')
"
    fi
done
