#!/bin/bash
# AG Yeast Stage 2 sweep v5 — expanded hyperparameter search.
#
# v4 findings (encoder-only, 50K seqs, lr=1e-4 to 5e-4):
#   - Best: lr=5e-4, val=0.558, test_random=0.707, snv_abs=0.738
#   - S2 overfits fast (peak epoch 3-9, then decline)
#   - Only encoder unfreezing tested; 50K seqs is <1% of 6M
#
# This sweep explores:
#   1. More training data (100K sequences) to reduce overfitting
#   2. Backbone unfreezing (transformer_tower + decoder) for cross-species adaptation
#   3. Gradual unfreezing strategy
#   4. Higher weight decay to regularize
#   5. No-shift augmentation (RC only) as contrast
#
# 12 configs as array job. 50K configs ~8-12h, 100K configs ~15-20h (48h wall time).
#
# Speed optimizations vs v4:
#   - batch_size 256→1024 (yeast T=3 tokens; bs=256 severely underutilizes H100)
#   - early_stop_patience 10→7 (S2 peaks at epoch 3-4 then declines)
#   - eval_use_reverse_complement=false during sweep (halves val cost; only
#     need relative ranking, not absolute numbers)
#
# Submit:
#   /cm/shared/apps/slurm/current/bin/sbatch scripts/slurm/ag_yeast_s2_v5.sh
#
#SBATCH --job-name=ag_yeast_s2_v5
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=200G
#SBATCH --array=0-11

set -euo pipefail

set +u
source /etc/profile.d/modules.sh
set -u
module load EB5

cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
source scripts/slurm/setup_hpc_deps.sh


OUT_BASE="outputs/ag_yeast_sweep_s2_v5"

# Common S1 + S2 base args
BASE_ARGS=(
  "++wandb_mode=offline"
  "++batch_size=4096"
  "++lr=0.003"
  "++epochs=5"
  "++early_stop_patience=100"
  "++second_stage_epochs=50"
  "++second_stage_batch_size=1024"
  "++second_stage_early_stop_patience=7"
  "++eval_use_reverse_complement=false"
)

# === Core grid: 2 max_seqs × 2 unfreeze × 2 lr = 8 configs ===
# === Extended: 4 targeted configs ===
case "${SLURM_ARRAY_TASK_ID}" in
  # --- Core grid: 50K sequences ---
  0)
    TAG="50k_enc_lr3e4"
    EXTRA_ARGS=(
      "++output_dir=${OUT_BASE}/${TAG}"
      "++second_stage_lr=3e-4"
      "++second_stage_weight_decay=1e-6"
      "++second_stage_max_sequences=50000"
      "++second_stage_unfreeze_mode=encoder"
    ) ;;
  1)
    TAG="50k_enc_lr5e4"
    EXTRA_ARGS=(
      "++output_dir=${OUT_BASE}/${TAG}"
      "++second_stage_lr=5e-4"
      "++second_stage_weight_decay=1e-6"
      "++second_stage_max_sequences=50000"
      "++second_stage_unfreeze_mode=encoder"
    ) ;;
  2)
    TAG="50k_bb_lr3e4"
    EXTRA_ARGS=(
      "++output_dir=${OUT_BASE}/${TAG}"
      "++second_stage_lr=3e-4"
      "++second_stage_weight_decay=1e-6"
      "++second_stage_max_sequences=50000"
      "++second_stage_unfreeze_mode=backbone"
    ) ;;
  3)
    TAG="50k_bb_lr5e4"
    EXTRA_ARGS=(
      "++output_dir=${OUT_BASE}/${TAG}"
      "++second_stage_lr=5e-4"
      "++second_stage_weight_decay=1e-6"
      "++second_stage_max_sequences=50000"
      "++second_stage_unfreeze_mode=backbone"
    ) ;;
  # --- Core grid: 100K sequences ---
  4)
    TAG="100k_enc_lr3e4"
    EXTRA_ARGS=(
      "++output_dir=${OUT_BASE}/${TAG}"
      "++second_stage_lr=3e-4"
      "++second_stage_weight_decay=1e-6"
      "++second_stage_max_sequences=100000"
      "++second_stage_unfreeze_mode=encoder"
    ) ;;
  5)
    TAG="100k_enc_lr5e4"
    EXTRA_ARGS=(
      "++output_dir=${OUT_BASE}/${TAG}"
      "++second_stage_lr=5e-4"
      "++second_stage_weight_decay=1e-6"
      "++second_stage_max_sequences=100000"
      "++second_stage_unfreeze_mode=encoder"
    ) ;;
  6)
    TAG="100k_bb_lr3e4"
    EXTRA_ARGS=(
      "++output_dir=${OUT_BASE}/${TAG}"
      "++second_stage_lr=3e-4"
      "++second_stage_weight_decay=1e-6"
      "++second_stage_max_sequences=100000"
      "++second_stage_unfreeze_mode=backbone"
    ) ;;
  7)
    TAG="100k_bb_lr5e4"
    EXTRA_ARGS=(
      "++output_dir=${OUT_BASE}/${TAG}"
      "++second_stage_lr=5e-4"
      "++second_stage_weight_decay=1e-6"
      "++second_stage_max_sequences=100000"
      "++second_stage_unfreeze_mode=backbone"
    ) ;;
  # --- Extended: targeted configs ---
  8)
    # Higher weight decay to combat overfitting
    TAG="100k_bb_lr5e4_wd1e4"
    EXTRA_ARGS=(
      "++output_dir=${OUT_BASE}/${TAG}"
      "++second_stage_lr=5e-4"
      "++second_stage_weight_decay=1e-4"
      "++second_stage_max_sequences=100000"
      "++second_stage_unfreeze_mode=backbone"
    ) ;;
  9)
    # Gradual unfreezing: encoder first, then backbone at epoch 5
    TAG="100k_gradual_lr5e4"
    EXTRA_ARGS=(
      "++output_dir=${OUT_BASE}/${TAG}"
      "++second_stage_lr=5e-4"
      "++second_stage_weight_decay=1e-6"
      "++second_stage_max_sequences=100000"
      "++second_stage_unfreeze_mode=gradual"
      "++second_stage_full_unfreeze_epoch=5"
    ) ;;
  10)
    # No shift augmentation (RC only) — test if shift hurts on short yeast seqs
    TAG="100k_bb_lr5e4_noshift"
    EXTRA_ARGS=(
      "++output_dir=${OUT_BASE}/${TAG}"
      "++second_stage_lr=5e-4"
      "++second_stage_weight_decay=1e-6"
      "++second_stage_max_sequences=100000"
      "++second_stage_unfreeze_mode=backbone"
      "++second_stage_max_shift=0"
    ) ;;
  11)
    # Higher LR to match 4x larger batch size (linear scaling from v4's 5e-4 at bs=256)
    TAG="100k_bb_lr1e3"
    EXTRA_ARGS=(
      "++output_dir=${OUT_BASE}/${TAG}"
      "++second_stage_lr=1e-3"
      "++second_stage_weight_decay=1e-6"
      "++second_stage_max_sequences=100000"
      "++second_stage_unfreeze_mode=backbone"
    ) ;;
  *)
    echo "Unexpected SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
    exit 1 ;;
esac

echo "=== AG yeast S2 v5: ${TAG} (task ${SLURM_ARRAY_TASK_ID}) ==="
echo "Node: ${SLURMD_NODENAME}  Date: $(date)"
START_TIME=$(date +%s)

# Skip if result already exists
if [ -f "${OUT_BASE}/${TAG}/summary.json" ]; then
    echo "SKIP: ${TAG} already has summary.json"
    exit 0
fi

uv run --no-sync python experiments/train_oracle_alphagenome_yeast.py \
  --config-name oracle_alphagenome_yeast_finetune_sweep \
  "${BASE_ARGS[@]}" \
  "${EXTRA_ARGS[@]}"

END_TIME=$(date +%s)
echo "=== ${TAG} DONE at $(date) — wall time: $((END_TIME - START_TIME))s ==="
