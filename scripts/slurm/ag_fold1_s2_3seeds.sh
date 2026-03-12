#!/bin/bash
# AlphaGenome fold_1 Stage 2: 3 seeds with all-folds best config (s2c).
#
# Uses the same hyperparameters as stage2_k562_full_train (all-folds S2):
#   encoder_lr=1e-4, head_lr=1e-3, unfreeze=[4,5], batch_size=128
#
# Requires fold1 S1 checkpoint to already exist at:
#   outputs/ag_fold1_s1_cached/fraction_1.0000/run_*/best_model/checkpoint
#
# Submit:
#   /cm/shared/apps/slurm/current/bin/sbatch scripts/slurm/ag_fold1_s2_3seeds.sh
#
#SBATCH --job-name=ag_fold1_s2
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=default
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=200G
#SBATCH --array=0-2

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
S2_OUTPUT="outputs/stage2_k562_fold1"

# Find S1 checkpoint
S1_RUN=$(find "$S1_OUTPUT/fraction_1.0000" -maxdepth 1 -name "run_*" -type d 2>/dev/null | sort | tail -1 || true)
if [ -z "$S1_RUN" ] || [ ! -d "$S1_RUN/best_model/checkpoint" ]; then
    echo "ERROR: S1 checkpoint not found in $S1_OUTPUT/fraction_1.0000/"
    exit 1
fi
echo "S1 checkpoint: $S1_RUN"

IDX=${SLURM_ARRAY_TASK_ID}
OUT_DIR="${S2_OUTPUT}/run_${IDX}"

echo "=== AG fold1 S2: run_${IDX} ==="
echo "Node: ${SLURMD_NODENAME}  Date: $(date)"

# Skip if already complete
if [ -f "${OUT_DIR}/test_metrics.json" ]; then
    echo "SKIP: result already exists at ${OUT_DIR}/test_metrics.json"
    exit 0
fi

uv run --no-sync python experiments/train_stage2_k562_hashfrag.py \
    --config-name stage2_k562_full_train \
    ++weights_path="$FOLD1_WEIGHTS" \
    ++stage1_dir="$S1_RUN" \
    ++output_dir="$OUT_DIR" \
    ++wandb_mode=offline

echo ""
echo "=== run_${IDX} COMPLETE — $(date) ==="
if [ -f "${OUT_DIR}/test_metrics.json" ]; then
    python3 -c "
import json
m = json.load(open('${OUT_DIR}/test_metrics.json'))
for k in ['in_distribution', 'snv_abs', 'snv_delta', 'ood']:
    if k in m:
        print(f'  {k}: {m[k][\"pearson_r\"]:.4f}')
"
fi
