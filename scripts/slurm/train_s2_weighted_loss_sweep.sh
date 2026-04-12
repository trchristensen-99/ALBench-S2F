#!/bin/bash
# S2 fine-tuning sweep: weighted loss variants for negative sequences.
#
# Tests 7 conditions on fold_0 only:
#   1. mse_1x   — standard MSE (baseline, neg_weight=1.0)
#   2. mse_2x   — weighted MSE, negatives × 2
#   3. mse_5x   — weighted MSE, negatives × 5
#   4. mse_10x  — weighted MSE, negatives × 10
#   5. focal_1  — focal-style MSE, gamma=1.0
#   6. focal_2  — focal-style MSE, gamma=2.0
#   7. huber_2x — Huber loss for negatives (delta=1.0), weight × 2
#
# Array: 0-6 (one task per condition)
#
# Submit:
#   /cm/shared/apps/slurm/current/bin/sbatch --array=0-6 \
#       scripts/slurm/train_s2_weighted_loss_sweep.sh
#
#SBATCH --job-name=s2_wloss
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=default
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

FOLD=0
S1_DIR="outputs/ag_hashfrag_oracle_cached/oracle_0"

# ── Condition lookup ──────────────────────────────────────────────────────────
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}

case "${TASK_ID}" in
    0)
        LOSS_TYPE="mse"
        NEG_WEIGHT="1.0"
        FOCAL_GAMMA="1.0"
        HUBER_DELTA="1.0"
        NAME="mse_1x"
        ;;
    1)
        LOSS_TYPE="mse"
        NEG_WEIGHT="2.0"
        FOCAL_GAMMA="1.0"
        HUBER_DELTA="1.0"
        NAME="mse_2x"
        ;;
    2)
        LOSS_TYPE="mse"
        NEG_WEIGHT="5.0"
        FOCAL_GAMMA="1.0"
        HUBER_DELTA="1.0"
        NAME="mse_5x"
        ;;
    3)
        LOSS_TYPE="mse"
        NEG_WEIGHT="10.0"
        FOCAL_GAMMA="1.0"
        HUBER_DELTA="1.0"
        NAME="mse_10x"
        ;;
    4)
        LOSS_TYPE="focal"
        NEG_WEIGHT="1.0"
        FOCAL_GAMMA="1.0"
        HUBER_DELTA="1.0"
        NAME="focal_g1"
        ;;
    5)
        LOSS_TYPE="focal"
        NEG_WEIGHT="1.0"
        FOCAL_GAMMA="2.0"
        HUBER_DELTA="1.0"
        NAME="focal_g2"
        ;;
    6)
        LOSS_TYPE="huber_neg"
        NEG_WEIGHT="2.0"
        FOCAL_GAMMA="1.0"
        HUBER_DELTA="1.0"
        NAME="huber_2x"
        ;;
    *)
        echo "ERROR: unknown TASK_ID=${TASK_ID}"
        exit 1
        ;;
esac

OUT_DIR="outputs/oracle_neg_sweep/weighted_loss_${NAME}/fold_${FOLD}"

echo "=== S2 weighted-loss sweep: condition=${NAME}, fold=${FOLD} — $(date) ==="
echo "    loss_type=${LOSS_TYPE}, neg_weight=${NEG_WEIGHT}, focal_gamma=${FOCAL_GAMMA}"
echo "    output_dir=${OUT_DIR}"

# Skip if already done
if [ -f "${OUT_DIR}/test_metrics.json" ]; then
    echo "SKIP: already done"
    exit 0
fi

uv run --no-sync python scripts/train_s2_weighted_loss.py \
    --config-name stage2_k562_oracle \
    ++fold_id="${FOLD}" \
    ++n_folds=10 \
    ++stage1_dir="${S1_DIR}" \
    ++output_dir="${OUT_DIR}" \
    ++loss_type="${LOSS_TYPE}" \
    ++neg_weight="${NEG_WEIGHT}" \
    ++focal_gamma="${FOCAL_GAMMA}" \
    ++huber_delta="${HUBER_DELTA}" \
    ++wandb_mode=offline

echo "=== Condition ${NAME} DONE — $(date) ==="
