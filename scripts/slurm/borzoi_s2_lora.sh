#!/bin/bash
# Borzoi S2 v6: LoRA adapter approach.
#
# Instead of unfreezing encoder blocks (which fails due to gradient dilution
# from 99.7% zero-padding), add small trainable adapter layers between
# frozen transformer blocks. Only ~0.4-1.6M trainable params vs 126M for
# full unfreeze.
#
# Array:
#   0: LoRA rank=32, last 4 blocks, center 4 bins, lr=1e-3
#   1: LoRA rank=64, last 4 blocks, center 4 bins, lr=1e-3
#   2: LoRA rank=32, all 8 blocks, center 4 bins, lr=1e-3
#   3: LoRA rank=32, last 4 blocks, all bins, lr=1e-3
#   4: LoRA rank=32, last 4 blocks, center 20 bins, lr=5e-4
#
#SBATCH --job-name=borz_lora
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
echo "=== Borzoi S2 LoRA task=${T} node=${SLURMD_NODENAME} date=$(date) ==="

COMMON="++model_name=borzoi ++cell_line=k562 ++epochs=40 ++warmup_epochs=5 ++early_stop_patience=10 ++batch_size=4"

case ${T} in
0)
    echo "LoRA rank=32, last 4 blocks, center 4 bins, lr=1e-3"
    uv run --no-sync python experiments/train_foundation_stage2.py \
        ${COMMON} \
        ++output_dir="outputs/borzoi_k562_s2_lora/r32_last4_c4_lr1e-3" \
        ++seed=42 \
        ++encoder_lr=0.001 \
        ++borzoi_center_bins=4 \
        ++use_lora=True \
        ++lora_rank=32 \
        ++lora_blocks="4,5,6,7"
    ;;
1)
    echo "LoRA rank=64, last 4 blocks, center 4 bins, lr=1e-3"
    uv run --no-sync python experiments/train_foundation_stage2.py \
        ${COMMON} \
        ++output_dir="outputs/borzoi_k562_s2_lora/r64_last4_c4_lr1e-3" \
        ++seed=42 \
        ++encoder_lr=0.001 \
        ++borzoi_center_bins=4 \
        ++use_lora=True \
        ++lora_rank=64 \
        ++lora_blocks="4,5,6,7"
    ;;
2)
    echo "LoRA rank=32, all 8 blocks, center 4 bins, lr=1e-3"
    uv run --no-sync python experiments/train_foundation_stage2.py \
        ${COMMON} \
        ++output_dir="outputs/borzoi_k562_s2_lora/r32_all8_c4_lr1e-3" \
        ++seed=42 \
        ++encoder_lr=0.001 \
        ++borzoi_center_bins=4 \
        ++use_lora=True \
        ++lora_rank=32 \
        ++lora_blocks="0,1,2,3,4,5,6,7"
    ;;
3)
    echo "LoRA rank=32, last 4 blocks, all bins, lr=1e-3"
    uv run --no-sync python experiments/train_foundation_stage2.py \
        ${COMMON} \
        ++output_dir="outputs/borzoi_k562_s2_lora/r32_last4_allbins_lr1e-3" \
        ++seed=42 \
        ++encoder_lr=0.001 \
        ++borzoi_center_bins=0 \
        ++use_lora=True \
        ++lora_rank=32 \
        ++lora_blocks="4,5,6,7"
    ;;
4)
    echo "LoRA rank=32, last 4 blocks, center 20 bins, lr=5e-4"
    uv run --no-sync python experiments/train_foundation_stage2.py \
        ${COMMON} \
        ++output_dir="outputs/borzoi_k562_s2_lora/r32_last4_c20_lr5e-4" \
        ++seed=42 \
        ++encoder_lr=0.0005 \
        ++borzoi_center_bins=20 \
        ++use_lora=True \
        ++lora_rank=32 \
        ++lora_blocks="4,5,6,7"
    ;;
*)
    echo "ERROR: unknown task ${T}"
    exit 1
    ;;
esac

echo "=== Done: $(date) ==="
