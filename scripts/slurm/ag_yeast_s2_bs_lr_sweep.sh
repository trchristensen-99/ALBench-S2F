#!/bin/bash
# AG Yeast S2: batch size × LR interaction sweep.
#
# Purpose: Find optimal BS for final scaling experiments.
#   - Large BS (1024) = fast sweeps but may sacrifice quality
#   - Smaller BS (128-256) = slower but potentially better convergence
#   - LR must be co-tuned with BS
#
# Previous findings:
#   - lr>=1e-3 diverges at bs=256 (NaN)
#   - lr=3e-4 to 5e-4 is optimal at bs=256 (val~0.558)
#   - BS sweep at lr=1e-3 showed clear degradation with larger BS
#   - S2 peaks at epoch 3-4 → 15 S2 epochs is plenty
#
# Grid: 4 BS × 3 LR = 12 configs, 15 S2 epochs (fast)
# ~2-6h per config depending on batch size.
#
# Submit:
#   /cm/shared/apps/slurm/current/bin/sbatch scripts/slurm/ag_yeast_s2_bs_lr_sweep.sh
#
#SBATCH --job-name=ag_s2_bs_lr
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --time=12:00:00
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

export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_command_buffer= --xla_gpu_autotune_level=0"

OUT_BASE="outputs/ag_yeast_s2_bs_lr_sweep"

# Grid: 4 BS × 3 LR = 12 configs
BATCH_SIZES=(128 256 512 1024)
LRS=(1e-4 3e-4 5e-4)

N_BS=${#BATCH_SIZES[@]}
N_LR=${#LRS[@]}

BS_IDX=$((SLURM_ARRAY_TASK_ID % N_BS))
LR_IDX=$((SLURM_ARRAY_TASK_ID / N_BS))

BS=${BATCH_SIZES[$BS_IDX]}
LR=${LRS[$LR_IDX]}
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
