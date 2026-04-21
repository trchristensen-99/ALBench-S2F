#!/bin/bash
# Retrain AG S2 oracle: 10 folds in parallel.
#
# Each fold: Stage 1 head training (cached embeddings, ~5 min) +
# Stage 2 encoder fine-tuning (~30-45 min on H100).
#
# Uses the s2c config (encoder_lr=1e-4, head_lr=1e-3) which was
# the best from earlier sweeps.
#
# Array: 0-9 (one fold per GPU)
#
#SBATCH --job-name=s2_orc
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G

set -euo pipefail
set +u; source /etc/profile.d/modules.sh; set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
source scripts/slurm/setup_hpc_deps.sh
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_command_buffer="
export ALPHAGENOME_WEIGHTS="/grid/wsbs/home_norepl/christen/alphagenome_weights/alphagenome-jax-all_folds-v1"

FOLD=$SLURM_ARRAY_TASK_ID
OUT_DIR="outputs/stage2_k562_oracle/fold_${FOLD}"

[ -f "${OUT_DIR}/test_metrics.json" ] && echo "SKIP: fold ${FOLD} already done" && exit 0

echo "=== S2 Oracle Fold ${FOLD}/10 — $(date) ==="

# Stage 2 fine-tuning with dedicated val split
uv run --no-sync python experiments/train_stage2_k562_hashfrag.py \
    --config-name stage2_k562_full_train \
    variant=s2c \
    encoder_lr=1e-4 \
    head_lr=1e-3 \
    epochs=15 \
    ++batch_size=128 \
    output_dir="${OUT_DIR}" \
    ++fold_id="${FOLD}" \
    ++n_folds=10 \
    ++debias_mode=none \
    ++debias_lambda=0

echo "=== DONE — $(date) ==="
