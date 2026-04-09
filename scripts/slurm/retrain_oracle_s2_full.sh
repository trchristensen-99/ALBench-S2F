#!/bin/bash
# Stage 2 oracle: fine-tune AG encoder on full 856K dataset.
#
# Each fold starts from its S1 checkpoint and fine-tunes the encoder.
# Uses s2c config (encoder_lr=1e-4, head_lr=1e-3, downres blocks 4,5).
#
# Array: 0-9 (one per fold, all parallel)
#
# Prerequisites: S1 folds complete at outputs/oracle_full_856k/s1/oracle_{0..9}/
#
# Submit:
#   /cm/shared/apps/slurm/current/bin/sbatch --array=0-9 scripts/slurm/retrain_oracle_s2_full.sh
#
#SBATCH --job-name=orc_s2
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

FOLD=${SLURM_ARRAY_TASK_ID}
S1_DIR="outputs/oracle_full_856k/s1/oracle_${FOLD}"
OUT_DIR="outputs/oracle_full_856k/s2/fold_${FOLD}"

echo "=== S2 oracle fold ${FOLD} node=${SLURMD_NODENAME} $(date) ==="

# Check S1 exists
if [ ! -d "${S1_DIR}/best_model" ]; then
    echo "ERROR: S1 fold ${FOLD} not found at ${S1_DIR}"
    exit 1
fi

# Skip if done
if [ -f "${OUT_DIR}/test_metrics.json" ]; then
    echo "SKIP: already done"
    exit 0
fi

uv run --no-sync python experiments/train_stage2_k562_hashfrag.py \
    --config-name stage2_k562_oracle \
    ++fold_id="${FOLD}" \
    ++n_folds=10 \
    ++stage1_dir="${S1_DIR}" \
    ++output_dir="${OUT_DIR}" \
    ++use_full_dataset=True \
    ++wandb_mode=offline

# Clean up
rm -rf "${OUT_DIR}/last_model" 2>/dev/null
echo "=== Fold ${FOLD} DONE — $(date) ==="
