#!/bin/bash
# Exp 0: DREAM-RNN yeast scaling curve v2 (optimized HPs + best-model fix).
#
# v1 used default config (bs=1024, lr=0.005, no early stopping) and had a bug
# that evaluated the last epoch model instead of best_model.pt.
#
# v2 uses optimized HPs from grid search:
#   bs=512, lr=0.005, lr_lstm=0.005, dropout_lstm=0.3, dropout_cnn=0.2
#   epochs=30, early_stopping_patience=10
#
# Grid search showed bs=512 configs dominate (random Pearson 0.813-0.816 vs
# 0.803-0.805 for bs=128). lr=0.005/do=0.3 chosen as robust middle ground.
#
# 3 seeds × 10 fractions = 30 tasks.
#
# Submit:
#   /cm/shared/apps/slurm/current/bin/sbatch scripts/slurm/exp0_yeast_scaling_v2.sh
#
#SBATCH --job-name=exp0_yeast_v2
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --array=0-29

set -euo pipefail

set +u
source /etc/profile.d/modules.sh
set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
source scripts/slurm/setup_hpc_deps.sh

FRACTIONS=(0.001 0.002 0.005 0.01 0.02 0.05 0.10 0.20 0.50 1.00)
SEEDS=(42 123 456)
N_FRACTIONS=${#FRACTIONS[@]}

SEED_IDX=$(( SLURM_ARRAY_TASK_ID / N_FRACTIONS ))
FRAC_IDX=$(( SLURM_ARRAY_TASK_ID % N_FRACTIONS ))
FRACTION=${FRACTIONS[$FRAC_IDX]}
SEED=${SEEDS[$SEED_IDX]}

OUT_DIR="outputs/exp0_yeast_scaling_v2"

echo "=== DREAM-RNN yeast scaling v2: fraction=${FRACTION} seed=${SEED} (task ${SLURM_ARRAY_TASK_ID}) ==="
echo "Node: ${SLURMD_NODENAME}  Date: $(date)"
START_TIME=$(date +%s)

# Skip if result already exists
FRAC_FMT=$(printf "%.4f" "${FRACTION}")
if [ -f "${OUT_DIR}/seed_${SEED}/fraction_${FRAC_FMT}/result.json" ]; then
    echo "SKIP: result already exists"
    exit 0
fi

uv run --no-sync python experiments/exp0_yeast_scaling.py \
    fraction="${FRACTION}" \
    output_dir="${OUT_DIR}" \
    batch_size=512 \
    lr=0.005 \
    lr_lstm=0.005 \
    dropout_lstm=0.3 \
    dropout_cnn=0.2 \
    epochs=30 \
    early_stopping_patience=10 \
    seed="${SEED}" \
    num_workers=4 \
    test_subset_dir=data/yeast/test_subset_ids \
    wandb_mode=offline

END_TIME=$(date +%s)
echo "=== DONE — wall time: $((END_TIME - START_TIME))s ==="
