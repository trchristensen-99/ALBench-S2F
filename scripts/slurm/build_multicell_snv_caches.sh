#!/bin/bash
# Build SNV embedding caches and re-evaluate SNV metrics for foundation models
# on HepG2/SK-N-SH using cached embeddings (matching S1 training pipeline).
#
# The S1 heads were trained on float16 cached embeddings, but the original SNV
# re-evaluation used live encoder passes (float32), producing different embeddings
# that cause metric degradation. This script fixes that by building proper SNV
# caches and evaluating through the cached pipeline.
#
# Array job (6 tasks):
#   0: Borzoi   HepG2
#   1: Borzoi   SKNSH
#   2: Enformer HepG2
#   3: Enformer SKNSH
#   4: NTv3     HepG2
#   5: NTv3     SKNSH
#
# Each task builds the SNV cache (if not already built), then evaluates all
# 3 seeds for that model/cell combination.
#
# Submit:
#   /cm/shared/apps/slurm/current/bin/sbatch scripts/slurm/build_multicell_snv_caches.sh
#
#SBATCH --job-name=snv_cache
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=default
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=200G
#SBATCH --array=0-5

set -euo pipefail

set +u
source /etc/profile.d/modules.sh
set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
source scripts/slurm/setup_hpc_deps.sh

# Prevent CUDA command buffer issues on some GPU nodes
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_command_buffer="

# ── Task configurations ───────────────────────────────────────────────────────
# Format: MODEL CELL
CONFIGS=(
    "borzoi hepg2"
    "borzoi sknsh"
    "enformer hepg2"
    "enformer sknsh"
    "ntv3_post hepg2"
    "ntv3_post sknsh"
)

CFG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
read -r MODEL CELL <<< "$CFG"

echo "=== SNV Cache Build + Eval ==="
echo "Model: ${MODEL}, Cell: ${CELL}"
echo "Task: ${SLURM_ARRAY_TASK_ID}/5"
echo "Node: ${SLURMD_NODENAME}  Date: $(date)"

# ── Phase 1: Build SNV caches ────────────────────────────────────────────────
# Build once per model/cell. Cache is shared across seeds.
# The first seed run builds; subsequent runs skip (cache exists).

echo ""
echo "--- Phase 1: Build SNV embedding cache ---"

# Determine result dir for first seed (to test cache build)
FIRST_SEED_DIR="outputs/${MODEL}_${CELL}_cached/seed_0/seed_0"

uv run --no-sync python scripts/build_and_eval_snv_cache.py \
    --model "${MODEL}" \
    --cell-line "${CELL}" \
    --result-dir "${FIRST_SEED_DIR}" \
    --skip-eval

echo "Cache build done: $(date)"

# ── Phase 2: Evaluate all seeds ──────────────────────────────────────────────
echo ""
echo "--- Phase 2: Evaluate all seeds ---"

for SEED_IDX in 0 1 2; do
    RESULT_DIR="outputs/${MODEL}_${CELL}_cached/seed_${SEED_IDX}/seed_${SEED_IDX}"
    if [ ! -d "${RESULT_DIR}" ]; then
        echo "  SKIP seed_${SEED_IDX}: ${RESULT_DIR} does not exist"
        continue
    fi

    echo ""
    echo "--- Seed ${SEED_IDX} ---"
    uv run --no-sync python scripts/build_and_eval_snv_cache.py \
        --model "${MODEL}" \
        --cell-line "${CELL}" \
        --result-dir "${RESULT_DIR}" \
        --skip-cache
done

echo ""
echo "=== All done for ${MODEL} ${CELL}: $(date) ==="
