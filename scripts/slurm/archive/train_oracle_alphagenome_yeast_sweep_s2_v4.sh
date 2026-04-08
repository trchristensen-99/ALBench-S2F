#!/bin/bash
# Yeast AG Stage 2 sweep v4 — targeted LR range (1e-4 to 5e-4).
#
# v3 showed: lr=3e-4 works (val~0.558), lr=1e-3+ diverges, lr=1e-5 too slow.
# This sweep explores the sweet spot more finely.
#
# Grid (4 tasks):
#   0 → lr=1e-4, encoder, 5 S1 epochs
#   1 → lr=2e-4, encoder, 5 S1 epochs
#   2 → lr=5e-4, encoder, 5 S1 epochs
#   3 → lr=3e-4, encoder, 5 S1 epochs, BS=128 (test smaller BS)
#
# Note: lr=3e-4 at BS=256 is already running in v3 task 0.
#
# Submit:
#   /cm/shared/apps/slurm/current/bin/sbatch scripts/slurm/train_oracle_alphagenome_yeast_sweep_s2_v4.sh
#
#SBATCH --job-name=ag_yeast_s2_v4
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=200G
#SBATCH --array=0-3

set -euo pipefail

set +u
source /etc/profile.d/modules.sh
set -u
module load EB5

cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
source scripts/slurm/setup_hpc_deps.sh


OUT_BASE="outputs/ag_yeast_sweep_s2_v4"

case "${SLURM_ARRAY_TASK_ID}" in
  0)
    EXTRA_ARGS=(
      "++output_dir=${OUT_BASE}/s2_lr1e4_enc"
      "++second_stage_lr=1e-4"
      "++second_stage_batch_size=256"
    )
    ;;
  1)
    EXTRA_ARGS=(
      "++output_dir=${OUT_BASE}/s2_lr2e4_enc"
      "++second_stage_lr=2e-4"
      "++second_stage_batch_size=256"
    )
    ;;
  2)
    EXTRA_ARGS=(
      "++output_dir=${OUT_BASE}/s2_lr5e4_enc"
      "++second_stage_lr=5e-4"
      "++second_stage_batch_size=256"
    )
    ;;
  3)
    EXTRA_ARGS=(
      "++output_dir=${OUT_BASE}/s2_lr3e4_bs128"
      "++second_stage_lr=3e-4"
      "++second_stage_batch_size=128"
    )
    ;;
  *)
    echo "Unexpected SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
    exit 1
    ;;
esac

echo "=== S2 sweep v4 task=${SLURM_ARRAY_TASK_ID} ==="
echo "Node: ${SLURMD_NODENAME}  Date: $(date)"
START_TIME=$(date +%s)

uv run --no-sync python experiments/train_oracle_alphagenome_yeast.py \
  --config-name oracle_alphagenome_yeast_finetune_sweep \
  "++wandb_mode=offline" \
  "++batch_size=4096" \
  "++lr=0.003" \
  "++epochs=5" \
  "++early_stop_patience=100" \
  "++second_stage_epochs=50" \
  "++second_stage_weight_decay=1e-6" \
  "++second_stage_max_sequences=50000" \
  "++second_stage_early_stop_patience=10" \
  "++second_stage_unfreeze_mode=encoder" \
  "${EXTRA_ARGS[@]}"

END_TIME=$(date +%s)
echo "=== task ${SLURM_ARRAY_TASK_ID} DONE at $(date) — wall time: $((END_TIME - START_TIME))s ==="
