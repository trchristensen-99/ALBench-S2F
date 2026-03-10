#!/bin/bash
# AlphaGenome yeast head hyperparameter grid search (v2: array job).
# Sweeps lr × weight_decay × dropout on cached embeddings (27 configs).
# Uses f=1.0, seed=42, frozen encoder (cached).
#
# v1 was serial (all 27 configs in one job, ~14 min/epoch → too slow).
# v2: array job, one config per task → finishes in ~2-4h per config.
#
# Grid: 3 lr × 3 wd × 3 dropout = 27 configs
#   lr:      {0.0001, 0.0005, 0.001}
#   wd:      {1e-6, 1e-4, 1e-3}
#   dropout: {0.1, 0.3, 0.5}
#
# Submit:
#   /cm/shared/apps/slurm/current/bin/sbatch scripts/slurm/ag_yeast_head_grid_search.sh
#
#SBATCH --job-name=ag_yeast_grid
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=12:00:00
#SBATCH --array=0-26

set -euo pipefail

set +u
source /etc/profile.d/modules.sh
set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
source scripts/slurm/setup_hpc_deps.sh

export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_command_buffer= --xla_gpu_autotune_level=0"

OUT_BASE="outputs/foundation_grid_search/alphagenome"

# Grid: 3 lr × 3 wd × 3 dropout = 27 configs
LRS=(0.0001 0.0005 0.001)
WDS=(0.000001 0.0001 0.001)
DROPOUTS=(0.1 0.3 0.5)

N_LR=${#LRS[@]}
N_WD=${#WDS[@]}

LR_IDX=$((SLURM_ARRAY_TASK_ID / (N_WD * ${#DROPOUTS[@]})))
WD_IDX=$(((SLURM_ARRAY_TASK_ID / ${#DROPOUTS[@]}) % N_WD))
DO_IDX=$((SLURM_ARRAY_TASK_ID % ${#DROPOUTS[@]}))

LR=${LRS[$LR_IDX]}
WD=${WDS[$WD_IDX]}
DO=${DROPOUTS[$DO_IDX]}
TAG="lr${LR}_wd${WD}_do${DO}"
OUT_DIR="${OUT_BASE}/${TAG}"

echo "=== AG yeast head grid: ${TAG} (task ${SLURM_ARRAY_TASK_ID}/26) ==="
echo "Node: ${SLURMD_NODENAME}  Date: $(date)"
START_TIME=$(date +%s)

# Skip if result already exists
if ls "${OUT_DIR}"/fraction_*/seed_*/result.json > /dev/null 2>&1; then
    echo "SKIP: ${TAG} (already done)"
    exit 0
fi

uv run --no-sync python experiments/exp0_yeast_scaling_alphagenome.py \
    ++fraction=1.0 \
    ++seed=42 \
    ++output_dir="${OUT_DIR}" \
    ++lr="${LR}" \
    ++weight_decay="${WD}" \
    ++dropout_rate="${DO}" \
    ++epochs=50 \
    ++early_stop_patience=7 \
    ++wandb_mode=offline \
    ++test_subset_dir=data/yeast/test_subset_ids

END_TIME=$(date +%s)
echo "=== ${TAG} DONE at $(date) — wall time: $((END_TIME - START_TIME))s ==="
