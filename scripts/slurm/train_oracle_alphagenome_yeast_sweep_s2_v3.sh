#!/bin/bash
# Yeast AG Stage 2 encoder fine-tuning sweep v3 — HIGH learning rates.
#
# v2 used lr=1e-5/5e-6 which was far too conservative for yeast (cross-species
# transfer from human AlphaGenome requires substantial adaptation). This sweep
# explores LRs around 1e-3 which is much more appropriate.
#
# Grid (6 tasks):
#   0 → lr=3e-4, encoder unfreeze, 5 S1 epochs
#   1 → lr=1e-3, encoder unfreeze, 5 S1 epochs
#   2 → lr=3e-3, encoder unfreeze, 5 S1 epochs
#   3 → lr=1e-3, backbone unfreeze, 5 S1 epochs
#   4 → lr=1e-3, encoder unfreeze, 1 S1 epoch (less S1 pre-training)
#   5 → lr=1e-2, encoder unfreeze, 5 S1 epochs (aggressive)
#
# Speedups vs v2:
#   - batch_size=256 (vs 128) — ~2x throughput on H100 96GB
#   - max_sequences=50000 (vs 100000) — halves epoch time for faster iteration
#   - early_stop_patience=7 (faster pruning of bad configs)
#
# Submit:
#   /cm/shared/apps/slurm/current/bin/sbatch scripts/slurm/train_oracle_alphagenome_yeast_sweep_s2_v3.sh
#
#SBATCH --job-name=ag_yeast_s2_v3
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=200G
#SBATCH --array=0-5

set -euo pipefail

set +u
source /etc/profile.d/modules.sh
set -u
module load EB5

cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
source scripts/slurm/setup_hpc_deps.sh

export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_command_buffer= --xla_gpu_autotune_level=0"

# Stage 1: BS=4096 lr=3e-3 (optimal from BS×LR grid).
# Stage 2: BS=256, 50K sequences for faster iteration.
COMMON_ARGS=(
  "--config-name" "oracle_alphagenome_yeast_finetune_sweep"
  "++wandb_mode=offline"
  "++batch_size=4096"
  "++lr=0.003"
  "++second_stage_epochs=50"
  "++second_stage_batch_size=256"
  "++second_stage_weight_decay=1e-6"
  "++second_stage_max_sequences=50000"
  "++second_stage_early_stop_patience=7"
)

OUT_BASE="outputs/ag_yeast_sweep_s2_v3"

case "${SLURM_ARRAY_TASK_ID}" in
  0)
    # S2 lr=3e-4, encoder, 5 S1 epochs
    EXTRA_ARGS=(
      "++output_dir=${OUT_BASE}/s2_lr3e4_enc_s1ep5"
      "++epochs=5"
      "++early_stop_patience=100"
      "++second_stage_lr=3e-4"
      "++second_stage_unfreeze_mode=encoder"
    )
    ;;
  1)
    # S2 lr=1e-3, encoder, 5 S1 epochs
    EXTRA_ARGS=(
      "++output_dir=${OUT_BASE}/s2_lr1e3_enc_s1ep5"
      "++epochs=5"
      "++early_stop_patience=100"
      "++second_stage_lr=1e-3"
      "++second_stage_unfreeze_mode=encoder"
    )
    ;;
  2)
    # S2 lr=3e-3, encoder, 5 S1 epochs
    EXTRA_ARGS=(
      "++output_dir=${OUT_BASE}/s2_lr3e3_enc_s1ep5"
      "++epochs=5"
      "++early_stop_patience=100"
      "++second_stage_lr=3e-3"
      "++second_stage_unfreeze_mode=encoder"
    )
    ;;
  3)
    # S2 lr=1e-3, backbone (full), 5 S1 epochs
    EXTRA_ARGS=(
      "++output_dir=${OUT_BASE}/s2_lr1e3_backbone_s1ep5"
      "++epochs=5"
      "++early_stop_patience=100"
      "++second_stage_lr=1e-3"
      "++second_stage_unfreeze_mode=backbone"
    )
    ;;
  4)
    # S2 lr=1e-3, encoder, 1 S1 epoch (minimal S1 warmup)
    EXTRA_ARGS=(
      "++output_dir=${OUT_BASE}/s2_lr1e3_enc_s1ep1"
      "++epochs=1"
      "++early_stop_patience=100"
      "++second_stage_lr=1e-3"
      "++second_stage_unfreeze_mode=encoder"
    )
    ;;
  5)
    # S2 lr=1e-2, encoder, 5 S1 epochs (aggressive)
    EXTRA_ARGS=(
      "++output_dir=${OUT_BASE}/s2_lr1e2_enc_s1ep5"
      "++epochs=5"
      "++early_stop_patience=100"
      "++second_stage_lr=1e-2"
      "++second_stage_unfreeze_mode=encoder"
    )
    ;;
  *)
    echo "Unexpected SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
    exit 1
    ;;
esac

echo "=== S2 sweep v3 task=${SLURM_ARRAY_TASK_ID} ==="
echo "Node: ${SLURMD_NODENAME}  Date: $(date)"
START_TIME=$(date +%s)

uv run --no-sync python experiments/train_oracle_alphagenome_yeast.py \
  "${COMMON_ARGS[@]}" \
  "${EXTRA_ARGS[@]}"

END_TIME=$(date +%s)
echo "=== task ${SLURM_ARRAY_TASK_ID} DONE at $(date) — wall time: $((END_TIME - START_TIME))s ==="
