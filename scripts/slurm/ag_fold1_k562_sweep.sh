#!/bin/bash
# AlphaGenome fold_1 K562 S2 hyperparameter sweep
#
# 4-config grid: encoder_lr × unfreeze_blocks
#   0: encoder_lr=1e-4, unfreeze=[4,5]       (baseline = s2c)
#   1: encoder_lr=2e-4, unfreeze=[4,5]       (more aggressive encoder)
#   2: encoder_lr=1e-4, unfreeze=[2,3,4,5]   (more layers)
#   3: encoder_lr=2e-4, unfreeze=[2,3,4,5]   (both)
#
# Requires fold1 pipeline (cache + S1) to have completed first.
# Submit with dependency:
#   /cm/shared/apps/slurm/current/bin/sbatch --dependency=afterok:843447 scripts/slurm/ag_fold1_k562_sweep.sh
#
# Or submit standalone (if pipeline already done):
#   /cm/shared/apps/slurm/current/bin/sbatch scripts/slurm/ag_fold1_k562_sweep.sh
#
#SBATCH --job-name=ag_fold1_sweep
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --time=12:00:00
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

export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_command_buffer= --xla_gpu_autotune_level=0"
export PYTHONUNBUFFERED=1

FOLD1_WEIGHTS="/grid/wsbs/home_norepl/christen/alphagenome_weights/alphagenome-jax-fold_1"
S1_OUTPUT="outputs/ag_fold1_s1_cached"
SWEEP_OUTPUT="outputs/stage2_k562_fold1_sweep"

# Find the S1 checkpoint from the pipeline
S1_RUN=$(find "$S1_OUTPUT/fraction_1.0000" -maxdepth 1 -name "run_*" -type d | sort | tail -1)
if [ -z "$S1_RUN" ] || [ ! -d "$S1_RUN/best_model/checkpoint" ]; then
    echo "ERROR: S1 checkpoint not found. Run ag_fold1_k562_pipeline.sh first."
    exit 1
fi
echo "S1 checkpoint: $S1_RUN"

# Grid: encoder_lr × unfreeze_blocks
ENCODER_LRS=("0.0001" "0.0002" "0.0001" "0.0002")
UNFREEZE_BLOCKS=("4,5" "4,5" "2,3,4,5" "2,3,4,5")
LABELS=("elr1e-4_uf2" "elr2e-4_uf2" "elr1e-4_uf4" "elr2e-4_uf4")

IDX=${SLURM_ARRAY_TASK_ID}
ELR=${ENCODER_LRS[$IDX]}
UFB=${UNFREEZE_BLOCKS[$IDX]}
LABEL=${LABELS[$IDX]}
OUT_DIR="${SWEEP_OUTPUT}/${LABEL}"

echo "=== Fold1 S2 sweep: ${LABEL} (encoder_lr=${ELR}, unfreeze=[${UFB}]) ==="
echo "Node: ${SLURMD_NODENAME}  Date: $(date)"

# Skip if already complete
if [ -f "${OUT_DIR}/test_metrics.json" ]; then
    echo "SKIP: result already exists at ${OUT_DIR}/test_metrics.json"
    exit 0
fi

# Parse unfreeze_blocks into Hydra list format [2,3,4,5]
UFB_HYDRA="[${UFB}]"

uv run --no-sync python experiments/train_stage2_k562_hashfrag.py \
    --config-name stage2_k562_full_train \
    ++weights_path="$FOLD1_WEIGHTS" \
    ++stage1_dir="$S1_RUN" \
    ++output_dir="$OUT_DIR" \
    ++encoder_lr="$ELR" \
    ++unfreeze_encoder_blocks="$UFB_HYDRA" \
    ++wandb_mode=offline

echo ""
echo "=== Sweep config ${LABEL} COMPLETE — $(date) ==="
if [ -f "${OUT_DIR}/test_metrics.json" ]; then
    python3 -c "
import json
m = json.load(open('${OUT_DIR}/test_metrics.json'))
for k in ['in_distribution', 'snv_abs', 'snv_delta', 'ood']:
    if k in m:
        print(f'  {k}: {m[k][\"pearson_r\"]:.4f}')
"
fi
