#!/bin/bash
# Stage 2 oracle ensemble: 10-fold encoder fine-tuning on K562 hashFrag.
#
# Uses s2c hyperparameters (encoder_lr=1e-4, head_lr=1e-3).
# Each array task trains one fold (0-9), loading the corresponding
# Stage 1 oracle checkpoint and fine-tuning the encoder.
#
# Prerequisites:
#   All 10 Stage 1 oracle checkpoints must exist:
#     outputs/ag_hashfrag_oracle_cached/oracle_{0..9}/best_model/checkpoint
#
# Submit:
#   sbatch scripts/slurm/train_stage2_k562_oracle_array.sh
#
#SBATCH --job-name=ag_stage2_oracle
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=200G
#SBATCH --array=0-9

set -euo pipefail

FOLD_ID=${SLURM_ARRAY_TASK_ID}

set +u
source /etc/profile.d/modules.sh
set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
source scripts/slurm/setup_hpc_deps.sh

export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_command_buffer="

echo "Starting Stage 2 oracle fold ${FOLD_ID}/9 on $(date)"
echo "Node: ${SLURMD_NODENAME}"

OUT_DIR="outputs/stage2_k562_oracle/fold_${FOLD_ID}"

# Skip if already done
if [ -f "${OUT_DIR}/test_metrics.json" ]; then
    echo "SKIP: fold ${FOLD_ID} already done"
    exit 0
fi

uv run --no-sync python experiments/train_stage2_k562_hashfrag.py \
    --config-name stage2_k562_oracle \
    ++fold_id="${FOLD_ID}" \
    ++stage1_dir="outputs/ag_hashfrag_oracle_cached/oracle_${FOLD_ID}" \
    ++output_dir="${OUT_DIR}" \
    ++wandb_mode=offline

# Clean up last_model checkpoint to save disk space (best_model kept for pseudolabel gen)
rm -rf "${OUT_DIR}/last_model" 2>/dev/null
echo "Cleaned up last_model checkpoint to save disk space"
echo "=== Fold ${FOLD_ID} DONE — $(date) ==="
