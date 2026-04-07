#!/bin/bash
# Fix Enformer S2 seeds 1,2 for HepG2 and SknSh.
#
# Bug: fill_all_gaps.sh omitted ++data_path and ++stage1_result_dir for
# seeds 1,2, so they trained on K562 data with random head init.
# Result: in_dist OK (~0.84) but OOD collapsed (0.03-0.14 vs 0.54-0.59).
#
# Fix: Re-run with correct cell data + S1 warmstart, matching seed 0 config.
#
# Array:
#   0: HepG2 seed 1
#   1: HepG2 seed 2
#   2: SknSh seed 1
#   3: SknSh seed 2
#
# Submit:
#   /cm/shared/apps/slurm/current/bin/sbatch --array=0-3 scripts/slurm/fix_enformer_s2_hepg2_sknsh.sh
#
#SBATCH --job-name=enf_s2_fix
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --array=0-3

set -euo pipefail

set +u; source /etc/profile.d/modules.sh; set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
source scripts/slurm/setup_hpc_deps.sh

T=$SLURM_ARRAY_TASK_ID
echo "=== Enformer S2 fix task=${T} node=${SLURMD_NODENAME} $(date) ==="

CELLS=("hepg2" "hepg2" "sknsh" "sknsh")
SEEDS=(1 2 1 2)

CELL="${CELLS[$T]}"
SEED="${SEEDS[$T]}"

# Match seed 0's config: epochs=15, early_stop_patience=5
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

# Backup old (wrong) results
if [ -f "${OUT_DIR}/result.json" ]; then
    mv "${OUT_DIR}/result.json" "${OUT_DIR}/result_WRONG_k562.json"
fi
if [ -f "${OUT_DIR}/best_model.pt" ]; then
    mv "${OUT_DIR}/best_model.pt" "${OUT_DIR}/best_model_WRONG_k562.pt"
fi
if [ -f "${OUT_DIR}/test_predictions.npz" ]; then
    mv "${OUT_DIR}/test_predictions.npz" "${OUT_DIR}/test_predictions_WRONG_k562.npz"
fi

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
    ++batch_size=64 \
    ++save_encoder=True \
    ++amp_mode=bfloat16

echo "=== Done: task=${T} $(date) ==="
