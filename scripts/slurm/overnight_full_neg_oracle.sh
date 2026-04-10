#!/bin/bash
# Full 10-fold oracle training with best negative augmentation approach.
# Depends on results from overnight_neg_augmentation.sh (Step 3).
# If that hasn't run yet, defaults to random negatives at 10%.
#
# Array tasks 0-9: one per fold
#
# Submit after step 3 completes:
#   /cm/shared/apps/slurm/current/bin/sbatch --array=0-9 scripts/slurm/overnight_full_neg_oracle.sh
#
#SBATCH --job-name=neg_orc_full
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --time=06:00:00
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

FOLD=${SLURM_ARRAY_TASK_ID}
CACHE_DIR="outputs/oracle_full_856k/embedding_cache"
NEG_DIR="data/synthetic_negatives"

# Use 'all' negatives at 10% (or best from step 3)
NEG_TYPE="all"
NEG_FRAC="0.10"
OUT_DIR="outputs/oracle_neg_all_10pct/s1/oracle_${FOLD}"

echo "=== Oracle with negatives: fold=${FOLD}, neg=${NEG_TYPE}, frac=${NEG_FRAC} — $(date) ==="

if [ -f "${OUT_DIR}/test_metrics.json" ]; then
    echo "SKIP: already done"
    exit 0
fi

uv run --no-sync python scripts/train_oracle_s1_with_negatives.py \
    --cache-dir "${CACHE_DIR}" \
    --negatives-dir "${NEG_DIR}" \
    --neg-cache-dir "outputs/oracle_neg_augmentation/neg_embed_cache" \
    --output-dir "${OUT_DIR}" \
    --fold-id "${FOLD}" --n-folds 10 \
    --neg-type "${NEG_TYPE}" --neg-fraction "${NEG_FRAC}" \
    --epochs 50 --early-stop-patience 7 \
    --lr 0.001 --batch-size 128

echo "=== Fold ${FOLD} DONE — $(date) ==="
