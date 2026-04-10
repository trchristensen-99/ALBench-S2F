#!/bin/bash
# Stage 2 oracle fine-tuning WITH negative augmentation.
#
# Phase 1: Standard S2 training (encoder + head on original data)
# Phase 2: Continue training with negatives mixed in at lower LR
#
# Uses the existing S1 oracle checkpoints, then runs S2 with s2c config
# (encoder_lr=1e-4, head_lr=1e-3) for warmup, then adds negatives.
#
# Array: 0-9 (one per fold)
#
# Submit:
#   /cm/shared/apps/slurm/current/bin/sbatch --array=0-9 scripts/slurm/train_s2_with_negatives.sh
#
#SBATCH --job-name=s2_neg_aug
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

FOLD=${SLURM_ARRAY_TASK_ID}
S1_DIR="outputs/oracle_full_856k/s1/oracle_${FOLD}"
OUT_DIR="outputs/oracle_s2_neg_aug/fold_${FOLD}"

echo "=== S2 with negatives: fold=${FOLD} — $(date) ==="

# Skip if S1 checkpoint doesn't exist
if [ ! -d "${S1_DIR}/best_model" ]; then
    echo "ERROR: S1 fold ${FOLD} not found"
    exit 1
fi

# Skip if already done
if [ -f "${OUT_DIR}/test_metrics.json" ]; then
    echo "SKIP: already done"
    exit 0
fi

# Phase 1: Standard S2 training (uses existing train_stage2 script)
# s2c config: encoder_lr=1e-4, head_lr=1e-3
uv run --no-sync python experiments/train_stage2_k562_hashfrag.py \
    --config-name stage2_k562_oracle \
    ++fold_id="${FOLD}" \
    ++n_folds=10 \
    ++stage1_dir="${S1_DIR}" \
    ++output_dir="${OUT_DIR}" \
    ++use_full_dataset=True \
    ++include_negatives=True \
    ++negatives_path="data/synthetic_negatives/dinuc_shuffled_negatives.tsv" \
    ++neg_fraction=0.05 \
    ++wandb_mode=offline

echo "=== Fold ${FOLD} DONE — $(date) ==="
