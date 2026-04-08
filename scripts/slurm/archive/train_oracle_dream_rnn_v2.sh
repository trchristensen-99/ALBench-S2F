#!/bin/bash
# Train DREAM-RNN yeast oracle ensemble v2 (optimized HPs).
#
# v1 used default config (bs=1024, lr=0.005, dropout_lstm=0.5, 80 epochs,
# no early stopping). v2 uses optimized HPs from grid search.
#
# 10-fold cross-validation, 4 folds concurrent.
# Each fold trains on ~90% of 6M sequences → ~5.4M training samples.
# With bs=512 and 30 epochs: ~10,547 steps/epoch × 30 = ~4-6h per fold.
#
# Submit:
#   /cm/shared/apps/slurm/current/bin/sbatch scripts/slurm/train_oracle_dream_rnn_v2.sh
#
#SBATCH --job-name=oracle_dream_v2
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=24:00:00
#SBATCH --array=0-9%4

set -euo pipefail

set +u
source /etc/profile.d/modules.sh
set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
source scripts/slurm/setup_hpc_deps.sh

ORACLE_OUTPUT_ROOT="outputs/oracle_dream_rnn_yeast_kfold_v2"
OUT_DIR="${ORACLE_OUTPUT_ROOT}/oracle_${SLURM_ARRAY_TASK_ID}"

echo "=== Oracle DREAM-RNN v2: fold ${SLURM_ARRAY_TASK_ID}/9 ==="
echo "Node: ${SLURMD_NODENAME}  Date: $(date)"
START_TIME=$(date +%s)

# Skip if already done
if [ -f "${OUT_DIR}/summary.json" ]; then
    echo "SKIP: fold ${SLURM_ARRAY_TASK_ID} already done"
    exit 0
fi

uv run --no-sync python experiments/train_oracle_dream_rnn.py \
  ++data_path=data/yeast \
  ++n_folds=10 \
  ++fold_id="${SLURM_ARRAY_TASK_ID}" \
  ++fold_split_seed=42 \
  ++seed=null \
  ++output_dir="${OUT_DIR}" \
  ++batch_size=512 \
  ++lr=0.005 \
  ++lr_lstm=0.005 \
  ++dropout_lstm=0.3 \
  ++dropout_cnn=0.2 \
  ++epochs=30 \
  ++early_stopping_patience=10 \
  ++test_subset_dir=data/yeast/test_subset_ids \
  ++wandb_mode=offline

END_TIME=$(date +%s)
echo "=== Fold ${SLURM_ARRAY_TASK_ID} DONE — wall time: $((END_TIME - START_TIME))s ==="
