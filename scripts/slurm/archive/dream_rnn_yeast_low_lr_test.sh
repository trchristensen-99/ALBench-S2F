#!/bin/bash
# DREAM-RNN yeast: test low-LR hypothesis for small batch sizes.
#
# bs=512 beats bs=128 at every matched LR (0.003-0.01), but linear scaling
# rule suggests bs=128 should use ~4x lower LR. This tests whether bs=128
# or bs=256 with low LRs (0.001-0.002) can close the gap.
#
# 4 configs, seed=42, f=1.0. Purely informational — bs=512 is likely the
# practical choice for scaling experiments regardless.
#
# Submit:
#   /cm/shared/apps/slurm/current/bin/sbatch scripts/slurm/dream_rnn_yeast_low_lr_test.sh
#
#SBATCH --job-name=dream_yeast_lowlr
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-3

set -euo pipefail

set +u
source /etc/profile.d/modules.sh
set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
source scripts/slurm/setup_hpc_deps.sh

case "${SLURM_ARRAY_TASK_ID}" in
  0) BS=128; LR=0.001; DO=0.1 ;;
  1) BS=128; LR=0.002; DO=0.1 ;;
  2) BS=256; LR=0.001; DO=0.1 ;;
  3) BS=256; LR=0.002; DO=0.1 ;;
  *)
    echo "Unexpected SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
    exit 1 ;;
esac

TAG="bs${BS}_lr${LR}_do${DO}"
OUT_DIR="outputs/dream_yeast_low_lr_test/${TAG}"

echo "=== DREAM-RNN yeast low-LR test: ${TAG} (task ${SLURM_ARRAY_TASK_ID}) ==="
echo "Node: ${SLURMD_NODENAME}  Date: $(date)"
START_TIME=$(date +%s)

# Skip if result already exists
if [ -f "${OUT_DIR}/seed_42/scaling_curve.json" ]; then
    echo "SKIP: result already exists"
    exit 0
fi

uv run --no-sync python experiments/exp0_yeast_scaling.py \
    fraction=1.0 \
    output_dir="${OUT_DIR}" \
    batch_size="${BS}" \
    lr="${LR}" \
    lr_lstm="${LR}" \
    dropout_lstm="${DO}" \
    dropout_cnn=0.2 \
    epochs=30 \
    early_stopping_patience=10 \
    seed=42 \
    num_workers=4 \
    test_subset_dir=data/yeast/test_subset_ids \
    wandb_mode=offline

END_TIME=$(date +%s)
echo "=== ${TAG} DONE — wall time: $((END_TIME - START_TIME))s ==="
