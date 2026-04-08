#!/bin/bash
# Fix Enformer S2 OOM: gradient checkpointing + reduced batch size.
#
# The original fix_enformer_s2_hepg2_sknsh.sh used batch_size=64 which
# OOMs on H100 (93GB). This script uses:
#   - PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True (reduces fragmentation)
#   - gradient_checkpointing=True (recompute transformer activations in backward)
#   - batch_size=2, grad_accum_steps=4 (effective BS=8, fits in memory)
#   - --exclusive + --mem=200G for max memory headroom
#
# Array:
#   0: HepG2 seed 1
#   1: HepG2 seed 2
#   2: SknSh seed 1
#   3: SknSh seed 2
#
# Submit:
#   /cm/shared/apps/slurm/current/bin/sbatch --array=0-3 scripts/slurm/fix_enformer_s2_gradckpt.sh
#
#SBATCH --job-name=enf_s2_gc
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --exclusive
#SBATCH --array=0-3

set -euo pipefail

# Reduce CUDA memory fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

set +u; source /etc/profile.d/modules.sh; set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
source scripts/slurm/setup_hpc_deps.sh

T=$SLURM_ARRAY_TASK_ID
echo "=== Enformer S2 gradckpt fix task=${T} node=${SLURMD_NODENAME} $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

CELLS=("hepg2" "hepg2" "sknsh" "sknsh")
SEEDS=(1 2 1 2)

CELL="${CELLS[$T]}"
SEED="${SEEDS[$T]}"

OUT_DIR="outputs/enformer_${CELL}_stage2/seed_${SEED}"
S1_DIR="outputs/enformer_${CELL}_cached/seed_${SEED}/seed_${SEED}"

# Check S1 dir exists (might need different seed mapping)
if [ ! -d "${S1_DIR}" ]; then
    echo "WARNING: S1 dir not found at ${S1_DIR}"
    # Try seed 0 as fallback for S1 warmstart
    S1_DIR="outputs/enformer_${CELL}_cached/seed_0/seed_0"
    echo "Using S1 seed 0 as warmstart: ${S1_DIR}"
fi

echo "Cell: ${CELL}, Seed: ${SEED}"
echo "Output: ${OUT_DIR}"
echo "S1 warmstart: ${S1_DIR}"

# Backup old (wrong) results if they exist
for f in result.json best_model.pt test_predictions.npz; do
    if [ -f "${OUT_DIR}/${f}" ]; then
        mv "${OUT_DIR}/${f}" "${OUT_DIR}/${f%.${f##*.}}_OOM.${f##*.}"
    fi
done

uv run --no-sync python experiments/train_foundation_stage2.py \
    ++model_name=enformer \
    ++data_path="data/${CELL}" \
    ++cell_line="${CELL}" \
    ++chr_split=True \
    ++seed="${SEED}" \
    ++output_dir="${OUT_DIR}" \
    ++stage1_result_dir="${S1_DIR}" \
    ++encoder_lr=1e-4 \
    ++head_lr=1e-3 \
    ++epochs=15 \
    ++early_stop_patience=5 \
    ++unfreeze_mode=transformer \
    ++batch_size=2 \
    ++grad_accum_steps=4 \
    ++gradient_checkpointing=True \
    ++save_encoder=True \
    ++amp_mode=bfloat16

echo "=== Done: task=${T} $(date) ==="
