#!/bin/bash
# Yeast AG Stage 2 encoder fine-tuning sweep v2 (8 tasks).
# Updated: S1 uses BS=4096 lr=3e-3 (optimal from BS×LR grid).
# Stage 1: cached head-only training (~2-5 min with BS=4096).
# Stage 2: encoder fine-tuning on 100K sequence subset (~2.7h/epoch on H100).
#
# Submit:
#   /cm/shared/apps/slurm/current/bin/sbatch scripts/slurm/train_oracle_alphagenome_yeast_sweep_s2_v2.sh
#
#SBATCH --job-name=ag_yeast_s2_v2
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=200G
#SBATCH --array=0-7

set -euo pipefail

set +u
source /etc/profile.d/modules.sh
set -u
module load EB5

cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
source scripts/slurm/setup_hpc_deps.sh

export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_command_buffer= --xla_gpu_autotune_level=0"

# Stage 1: BS=4096 lr=3e-3 (optimal from BS×LR grid, nearly same val Pearson as BS=32).
# Stage 2: encoder fine-tuning at BS=128 (full encoder memory constraint).
COMMON_ARGS=(
  "--config-name" "oracle_alphagenome_yeast_finetune_sweep"
  "++wandb_mode=offline"
  "++batch_size=4096"
  "++lr=0.003"
  "++second_stage_epochs=50"
  "++second_stage_batch_size=128"
  "++second_stage_weight_decay=1e-6"
  "++second_stage_max_sequences=100000"
)

OUT_BASE="outputs/ag_yeast_sweep_s2_v2"

case "${SLURM_ARRAY_TASK_ID}" in
  0)
    # S2 baseline: full S1 early-stop, then encoder FT at lr=1e-5
    EXTRA_ARGS=(
      "++output_dir=${OUT_BASE}/s2_baseline_s1full_lr1e5"
      "++second_stage_lr=1e-5"
      "++second_stage_unfreeze_mode=encoder"
    )
    ;;
  1)
    # S2 after 1 S1 epoch
    EXTRA_ARGS=(
      "++output_dir=${OUT_BASE}/s2_s1ep1_lr1e5"
      "++epochs=1"
      "++early_stop_patience=100"
      "++second_stage_lr=1e-5"
      "++second_stage_unfreeze_mode=encoder"
    )
    ;;
  2)
    # S2 after 3 S1 epochs
    EXTRA_ARGS=(
      "++output_dir=${OUT_BASE}/s2_s1ep3_lr1e5"
      "++epochs=3"
      "++early_stop_patience=100"
      "++second_stage_lr=1e-5"
      "++second_stage_unfreeze_mode=encoder"
    )
    ;;
  3)
    # S2 after 5 S1 epochs
    EXTRA_ARGS=(
      "++output_dir=${OUT_BASE}/s2_s1ep5_lr1e5"
      "++epochs=5"
      "++early_stop_patience=100"
      "++second_stage_lr=1e-5"
      "++second_stage_unfreeze_mode=encoder"
    )
    ;;
  4)
    # S2 after 5 S1 epochs, full backbone unfreeze
    EXTRA_ARGS=(
      "++output_dir=${OUT_BASE}/s2_s1ep5_lr1e5_backbone"
      "++epochs=5"
      "++early_stop_patience=100"
      "++second_stage_lr=1e-5"
      "++second_stage_unfreeze_mode=backbone"
    )
    ;;
  5)
    # S2 after 5 S1 epochs, gradual unfreeze
    EXTRA_ARGS=(
      "++output_dir=${OUT_BASE}/s2_s1ep5_lr1e5_gradual"
      "++epochs=5"
      "++early_stop_patience=100"
      "++second_stage_lr=1e-5"
      "++second_stage_unfreeze_mode=gradual"
      "++second_stage_full_unfreeze_epoch=10"
    )
    ;;
  6)
    # S2 after 5 S1 epochs, lower s2_lr=5e-6
    EXTRA_ARGS=(
      "++output_dir=${OUT_BASE}/s2_s1ep5_lr5e6"
      "++epochs=5"
      "++early_stop_patience=100"
      "++second_stage_lr=5e-6"
      "++second_stage_unfreeze_mode=encoder"
    )
    ;;
  7)
    # S2 after 5 S1 epochs, no shift augmentation
    EXTRA_ARGS=(
      "++output_dir=${OUT_BASE}/s2_s1ep5_lr1e5_noshift"
      "++epochs=5"
      "++early_stop_patience=100"
      "++second_stage_lr=1e-5"
      "++second_stage_unfreeze_mode=encoder"
      "++second_stage_max_shift=0"
    )
    ;;
  *)
    echo "Unexpected SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
    exit 1
    ;;
esac

echo "=== S2 sweep v2 task=${SLURM_ARRAY_TASK_ID} ==="
echo "Node: ${SLURMD_NODENAME}  Date: $(date)"
START_TIME=$(date +%s)

uv run --no-sync python experiments/train_oracle_alphagenome_yeast.py \
  "${COMMON_ARGS[@]}" \
  "${EXTRA_ARGS[@]}"

END_TIME=$(date +%s)
echo "=== task ${SLURM_ARRAY_TASK_ID} DONE at $(date) — wall time: $((END_TIME - START_TIME))s ==="
