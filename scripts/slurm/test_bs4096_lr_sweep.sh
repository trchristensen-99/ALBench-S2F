#!/bin/bash
# LR optimization for BS=4096 cached head training.
# Baseline: BS=512, lr=1e-3. With 8x batch, try range of LR scaling strategies.
# Also includes BS=512 baseline for direct comparison.
#
# Submit:
#   /cm/shared/apps/slurm/current/bin/sbatch scripts/slurm/test_bs4096_lr_sweep.sh
#
#SBATCH --job-name=bs4096_lr
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=fast
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=200G
#SBATCH --array=0-6

set -euo pipefail

set +u
source /etc/profile.d/modules.sh
set -u
module load EB5

cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
source scripts/slurm/setup_hpc_deps.sh

export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_command_buffer= --xla_gpu_autotune_level=0"

OUT_BASE="outputs/bs4096_lr_sweep"

case "${SLURM_ARRAY_TASK_ID}" in
  0)
    # Baseline: BS=512, lr=1e-3 (reference)
    BS=512; LR=0.001; TAG="bs512_lr1e3_baseline"
    ;;
  1)
    # BS=4096, same lr=1e-3 (no scaling)
    BS=4096; LR=0.001; TAG="bs4096_lr1e3"
    ;;
  2)
    # BS=4096, sqrt scaling: lr=1e-3 * sqrt(8) ≈ 2.83e-3
    BS=4096; LR=0.00283; TAG="bs4096_lr2.8e3_sqrt"
    ;;
  3)
    # BS=4096, linear scaling: lr=1e-3 * 8 = 8e-3
    BS=4096; LR=0.008; TAG="bs4096_lr8e3_linear"
    ;;
  4)
    # BS=4096, lr=2e-3 (moderate)
    BS=4096; LR=0.002; TAG="bs4096_lr2e3"
    ;;
  5)
    # BS=4096, lr=4e-3
    BS=4096; LR=0.004; TAG="bs4096_lr4e3"
    ;;
  6)
    # BS=4096, lr=6e-3
    BS=4096; LR=0.006; TAG="bs4096_lr6e3"
    ;;
  *)
    echo "Unexpected SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
    exit 1
    ;;
esac

echo "=== BS=${BS} LR=${LR} (${TAG}) task=${SLURM_ARRAY_TASK_ID} ==="
echo "Node: ${SLURMD_NODENAME}  Date: $(date)"

uv run --no-sync python experiments/train_oracle_alphagenome_yeast.py \
  --config-name oracle_alphagenome_yeast_finetune_sweep \
  "++wandb_mode=offline" \
  "++batch_size=${BS}" \
  "++lr=${LR}" \
  "++epochs=20" \
  "++early_stop_patience=20" \
  "++second_stage_lr=null" \
  "++seed=42" \
  "++output_dir=${OUT_BASE}/${TAG}"

echo "=== ${TAG} DONE at $(date) ==="
