#!/bin/bash
# FULL Oracle Retraining Pipeline
#
# Step 1 (array task 10): Build embedding cache for ALL 856K sequences
# Step 2 (array tasks 0-9): Train S1 oracle heads (10-fold CV)
# Step 3 (separate job): Train S2 oracle (depends on S1 completion)
#
# The oracle is trained on ALL measured MPRA sequences:
#   - 798K ref sequences (full MPRA dataset, all chromosomes)
#   - 35K alt allele sequences (from SNV pairs)
#   - 23K OOD designed sequences
#   = 856K total, each with real K562 MPRA measurements
#
# 10-fold random CV: each fold trains on 90% (~770K), validates on 10% (~86K).
# Every sequence gets an out-of-fold prediction.
#
# Submit cache build first, then S1 folds:
#   CACHE_JOB=$(/cm/shared/apps/slurm/current/bin/sbatch --parsable --array=10 scripts/slurm/retrain_oracle_full.sh)
#   /cm/shared/apps/slurm/current/bin/sbatch --dependency=afterok:$CACHE_JOB --array=0-9 scripts/slurm/retrain_oracle_full.sh
#
# Or submit all at once (tasks 0-9 will wait for cache):
#   /cm/shared/apps/slurm/current/bin/sbatch --array=0-10 scripts/slurm/retrain_oracle_full.sh
#
#SBATCH --job-name=orc_full
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --time=12:00:00
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
CACHE_DIR="outputs/oracle_full_856k/embedding_cache"
S1_BASE="outputs/oracle_full_856k/s1"
DONE_FILE="${CACHE_DIR}/.cache_done"

echo "=== Oracle Full Retrain: task=${T} node=${SLURMD_NODENAME} $(date) ==="

if [ "${T}" -eq 10 ]; then
    # ════════════════════════════════════════════════════════════
    # STEP 1: Build embedding cache for ALL 856K sequences
    # ════════════════════════════════════════════════════════════
    echo "=== Building full embedding cache (856K sequences) ==="
    mkdir -p "${CACHE_DIR}"

    uv run --no-sync python scripts/build_full_oracle_cache.py \
        --output-dir "${CACHE_DIR}" \
        --batch-size 128

    touch "${DONE_FILE}"
    echo "=== Cache build complete: $(date) ==="

else
    # ════════════════════════════════════════════════════════════
    # STEP 2: Train S1 oracle for fold ${T}
    # ════════════════════════════════════════════════════════════
    FOLD=${T}

    # Wait for cache
    while [ ! -f "${DONE_FILE}" ]; do
        echo "Waiting for cache (fold ${FOLD})..."
        sleep 60
    done

    echo "=== Training S1 oracle fold ${FOLD} ==="
    OUT_DIR="${S1_BASE}/oracle_${FOLD}"
    mkdir -p "${OUT_DIR}"

    uv run --no-sync python scripts/train_oracle_s1_full.py \
        --cache-dir "${CACHE_DIR}" \
        --output-dir "${OUT_DIR}" \
        --fold-id "${FOLD}" \
        --n-folds 10 \
        --epochs 50 \
        --early-stop-patience 7 \
        --lr 0.001 \
        --batch-size 128

    echo "=== S1 fold ${FOLD} complete: $(date) ==="
fi
