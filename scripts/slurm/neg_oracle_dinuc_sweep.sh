#!/bin/bash
# Full 10-fold oracle with dinuc-shuffled negatives at 5%.
# Tests 3 label strategies:
#   A) Distribution: N(-0.454, 0.617) — current approach
#   B) Constant: all negatives get label = -0.454
#   C) Tight: N(-0.454, 0.274) — matches ENCODE raw std
#
# Array: 0-29 (3 strategies × 10 folds)
#
# Submit:
#   /cm/shared/apps/slurm/current/bin/sbatch --array=0-29 scripts/slurm/neg_oracle_dinuc_sweep.sh
#
#SBATCH --job-name=neg_dinuc_sweep
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --time=02:00:00
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

T=$SLURM_ARRAY_TASK_ID
FOLD=$((T % 10))
STRATEGY=$((T / 10))

CACHE_DIR="outputs/oracle_full_856k/embedding_cache"
NEG_CACHE="outputs/oracle_neg_augmentation/neg_embed_cache"

# Generate label variants if needed (only task 0 does this)
if [ "${T}" -eq 0 ]; then
    for VARIANT in const tight; do
        DIR="data/synthetic_negatives_${VARIANT}"
        if [ ! -f "${DIR}/dinuc_shuffled_negatives.tsv" ]; then
            echo "Generating ${VARIANT} label variant..."
            mkdir -p "${DIR}"
            uv run --no-sync python -c "
import csv, numpy as np
rng = np.random.default_rng(42)
seqs = []
with open('data/synthetic_negatives/dinuc_shuffled_negatives.tsv') as f:
    reader = csv.DictReader(f, delimiter='\t')
    for row in reader:
        seqs.append(row['sequence'])
mean_label = -0.4547
variant = '${VARIANT}'
with open('${DIR}/dinuc_shuffled_negatives.tsv', 'w', newline='') as f:
    w = csv.writer(f, delimiter='\t')
    w.writerow(['sequence', 'K562_log2FC', 'category'])
    for seq in seqs:
        if variant == 'const':
            label = mean_label
        else:  # tight
            label = rng.normal(mean_label, 0.274)
        w.writerow([seq, '%.6f' % label, 'dinuc_shuffled_negative'])
print('Generated %s: %d seqs' % (variant, len(seqs)))
" || true
        fi
    done
fi

# Wait for label generation (other tasks)
sleep 10

case $STRATEGY in
    0)
        LABEL="dist"
        NEG_DIR="data/synthetic_negatives"
        OUT_DIR="outputs/oracle_neg_dinuc_dist/s1/oracle_${FOLD}"
        ;;
    1)
        LABEL="const"
        NEG_DIR="data/synthetic_negatives_const"
        OUT_DIR="outputs/oracle_neg_dinuc_const/s1/oracle_${FOLD}"
        ;;
    2)
        LABEL="tight"
        NEG_DIR="data/synthetic_negatives_tight"
        OUT_DIR="outputs/oracle_neg_dinuc_tight/s1/oracle_${FOLD}"
        ;;
esac

echo "=== Strategy=${LABEL} fold=${FOLD} — $(date) ==="

if [ -f "${OUT_DIR}/test_metrics.json" ]; then
    echo "SKIP: already done"
    exit 0
fi

uv run --no-sync python scripts/train_oracle_s1_with_negatives.py \
    --cache-dir "${CACHE_DIR}" \
    --negatives-dir "${NEG_DIR}" \
    --neg-cache-dir "${NEG_CACHE}" \
    --output-dir "${OUT_DIR}" \
    --fold-id "${FOLD}" --n-folds 10 \
    --neg-type dinuc_shuffled \
    --neg-fraction 0.05 \
    --epochs 50 --early-stop-patience 7 \
    --lr 0.001 --batch-size 128

echo "=== Done: strategy=${LABEL} fold=${FOLD} — $(date) ==="
