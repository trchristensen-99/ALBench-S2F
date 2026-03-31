#!/bin/bash
# Borzoi S2 LoRA v2: Loss clipping + tighter grad clip fixes.
#
# v1 LoRA showed the adapters WERE learning (batch loss 0.1-0.5) but
# occasional catastrophic batches (loss 10-16) corrupted weights → NaN val.
#
# Fixes in train_foundation_stage2.py:
#   1. Loss clipping: skip batches with loss > 5.0 (prevents gradient explosions)
#   2. Extra tight grad clip (0.1) on LoRA adapter params specifically
#
# Also: lower LR (5e-4 instead of 1e-3) and longer warmup (10 epochs).
#
# Array:
#   0: LoRA rank=32, last 4 blocks, center 4 bins, lr=5e-4 (conservative)
#   1: LoRA rank=64, last 4 blocks, center 4 bins, lr=5e-4
#   2: LoRA rank=32, all 8 blocks, center 4 bins, lr=5e-4
#   3: LoRA rank=32, last 4 blocks, center 20 bins, lr=5e-4
#   4: LoRA rank=32, last 4 blocks, center 4 bins, lr=1e-4 (very conservative)
#
#SBATCH --job-name=borz_lr2
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=200G

set -euo pipefail

set +u; source /etc/profile.d/modules.sh; set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
source scripts/slurm/setup_hpc_deps.sh

T=$SLURM_ARRAY_TASK_ID
echo "=== Borzoi S2 LoRA v2 task=${T} node=${SLURMD_NODENAME} date=$(date) ==="

COMMON="++model_name=borzoi ++cell_line=k562 ++epochs=50 ++warmup_epochs=10 ++early_stop_patience=15 ++batch_size=4 ++grad_clip=0.5 ++use_lora=True"

case ${T} in
0)
    echo "LoRA r32, last4, c4, lr=5e-4"
    uv run --no-sync python experiments/train_foundation_stage2.py \
        ${COMMON} \
        ++output_dir="outputs/borzoi_k562_s2_lora_v2/r32_last4_c4_lr5e-4" \
        ++seed=42 ++encoder_lr=0.0005 \
        ++borzoi_center_bins=4 ++lora_rank=32 ++lora_blocks="4,5,6,7"
    ;;
1)
    echo "LoRA r64, last4, c4, lr=5e-4"
    uv run --no-sync python experiments/train_foundation_stage2.py \
        ${COMMON} \
        ++output_dir="outputs/borzoi_k562_s2_lora_v2/r64_last4_c4_lr5e-4" \
        ++seed=42 ++encoder_lr=0.0005 \
        ++borzoi_center_bins=4 ++lora_rank=64 ++lora_blocks="4,5,6,7"
    ;;
2)
    echo "LoRA r32, all8, c4, lr=5e-4"
    uv run --no-sync python experiments/train_foundation_stage2.py \
        ${COMMON} \
        ++output_dir="outputs/borzoi_k562_s2_lora_v2/r32_all8_c4_lr5e-4" \
        ++seed=42 ++encoder_lr=0.0005 \
        ++borzoi_center_bins=4 ++lora_rank=32 ++lora_blocks="0,1,2,3,4,5,6,7"
    ;;
3)
    echo "LoRA r32, last4, c20, lr=5e-4"
    uv run --no-sync python experiments/train_foundation_stage2.py \
        ${COMMON} \
        ++output_dir="outputs/borzoi_k562_s2_lora_v2/r32_last4_c20_lr5e-4" \
        ++seed=42 ++encoder_lr=0.0005 \
        ++borzoi_center_bins=20 ++lora_rank=32 ++lora_blocks="4,5,6,7"
    ;;
4)
    echo "LoRA r32, last4, c4, lr=1e-4 (very conservative)"
    uv run --no-sync python experiments/train_foundation_stage2.py \
        ${COMMON} \
        ++output_dir="outputs/borzoi_k562_s2_lora_v2/r32_last4_c4_lr1e-4" \
        ++seed=42 ++encoder_lr=0.0001 \
        ++borzoi_center_bins=4 ++lora_rank=32 ++lora_blocks="4,5,6,7"
    ;;
*)
    echo "ERROR: unknown task ${T}"
    exit 1
    ;;
esac

echo "=== Done: $(date) ==="
