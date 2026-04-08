#!/bin/bash
# AG Yeast S2: targeted batch size × LR interaction sweep.
#
# Purpose: Find optimal BS for final scaling experiments.
#   - LR must be co-tuned with BS (linear scaling rule)
#   - Smaller BS needs smaller LR; larger BS tolerates higher LR
#
# Previous findings:
#   - lr>=1e-3 diverges at bs=256 (NaN)
#   - lr=3e-4 to 5e-4 is optimal at bs=256 (val~0.558)
#   - bs=128 + lr=3e-4 gave single best val (0.5588)
#   - BS=1024 + lr=1e-3 was terrible (0.274)
#   - S2 peaks at epoch 3-4 → 15 S2 epochs is plenty
#
# Targeted grid (not full cross-product — skip known-bad corners):
#   BS=64:  lr={5e-5, 1e-4, 2e-4}      (small BS, small LR)
#   BS=128: lr={1e-4, 2e-4, 3e-4}      (best region from v4)
#   BS=256: lr={2e-4, 3e-4, 5e-4}      (well-characterized)
#   BS=512: lr={3e-4, 5e-4}            (moderate BS, moderate LR)
# = 11 configs total, 15 S2 epochs each
#
# Wall time: BS=64 is slowest (~6-8h), BS=512 fastest (~2h).
# 24h covers worst case with margin.
#
# Submit:
#   /cm/shared/apps/slurm/current/bin/sbatch scripts/slurm/ag_yeast_s2_bs_lr_sweep.sh
#
#SBATCH --job-name=ag_s2_bs_lr
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=200G
#SBATCH --array=0-10

set -euo pipefail

set +u
source /etc/profile.d/modules.sh
set -u
module load EB5

cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
source scripts/slurm/setup_hpc_deps.sh


OUT_BASE="outputs/ag_yeast_s2_bs_lr_sweep"

# Targeted grid: 11 configs (skip known-bad corners)
case "${SLURM_ARRAY_TASK_ID}" in
  # BS=64: small LRs
  0) BS=64;  LR=5e-5 ;;
  1) BS=64;  LR=1e-4 ;;
  2) BS=64;  LR=2e-4 ;;
  # BS=128: the sweet spot from v4
  3) BS=128; LR=1e-4 ;;
  4) BS=128; LR=2e-4 ;;
  5) BS=128; LR=3e-4 ;;
  # BS=256: well-characterized range
  6) BS=256; LR=2e-4 ;;
  7) BS=256; LR=3e-4 ;;
  8) BS=256; LR=5e-4 ;;
  # BS=512: moderate LRs only
  9)  BS=512; LR=3e-4 ;;
  10) BS=512; LR=5e-4 ;;
  *)
    echo "Unexpected SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
    exit 1 ;;
esac

TAG="bs${BS}_lr${LR}"

echo "=== AG S2 BS×LR sweep: ${TAG} (task ${SLURM_ARRAY_TASK_ID}) ==="
echo "Node: ${SLURMD_NODENAME}  Date: $(date)"
START_TIME=$(date +%s)

# Skip if result already exists
if [ -f "${OUT_BASE}/${TAG}/summary.json" ]; then
    echo "SKIP: ${TAG} already has summary.json"
    exit 0
fi

uv run --no-sync python experiments/train_oracle_alphagenome_yeast.py \
  --config-name oracle_alphagenome_yeast_finetune_sweep \
  "++wandb_mode=offline" \
  "++batch_size=4096" \
  "++lr=0.003" \
  "++epochs=5" \
  "++early_stop_patience=100" \
  "++second_stage_epochs=15" \
  "++second_stage_batch_size=${BS}" \
  "++second_stage_lr=${LR}" \
  "++second_stage_weight_decay=1e-6" \
  "++second_stage_max_sequences=50000" \
  "++second_stage_early_stop_patience=7" \
  "++second_stage_unfreeze_mode=encoder" \
  "++eval_use_reverse_complement=false" \
  "++output_dir=${OUT_BASE}/${TAG}"

END_TIME=$(date +%s)
echo "=== ${TAG} DONE at $(date) — wall time: $((END_TIME - START_TIME))s ==="
