#!/bin/bash
# BS×LR grid for AlphaGenome cached head on K562 (hashFrag, RC aug).
# Tests BS ∈ {128, 512, 1024, 4096, 16384} × LR ∈ {3e-4, 1e-3, 3e-3, 8e-3, 1.5e-2}
# = 25 configs. Uses fraction=1.0, 20 epochs, RC augmentation.
# Default: BS=512, lr=0.001 (boda-flatten-512-512, num_tracks=1).
# K562 has test embedding cache → test eval is fast (head-only, seconds).
#
# Submit:
#   /cm/shared/apps/slurm/current/bin/sbatch scripts/slurm/bs_lr_grid_ag_k562.sh
#
#SBATCH --job-name=bs_lr_ag_k562
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=200G
#SBATCH --array=0-24

set -euo pipefail

set +u
source /etc/profile.d/modules.sh
set -u
module load EB5

cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
source scripts/slurm/setup_hpc_deps.sh

export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_command_buffer= --xla_gpu_autotune_level=0"

# Grid: 5 batch sizes × 5 learning rates = 25 configs
BATCH_SIZES=(128 512 1024 4096 16384)
LRS=(0.0003 0.001 0.003 0.008 0.015)
N_LRS=${#LRS[@]}

BS_IDX=$((SLURM_ARRAY_TASK_ID / N_LRS))
LR_IDX=$((SLURM_ARRAY_TASK_ID % N_LRS))
BS=${BATCH_SIZES[$BS_IDX]}
LR=${LRS[$LR_IDX]}
TAG="bs${BS}_lr${LR}"

echo "=== AG K562 BS=${BS} LR=${LR} (${TAG}) task=${SLURM_ARRAY_TASK_ID} ==="
echo "Node: ${SLURMD_NODENAME}  Date: $(date)"
START_TIME=$(date +%s)

uv run --no-sync python experiments/exp0_k562_scaling_alphagenome_cached.py \
    --config-name exp0_k562_scaling_alphagenome_cached_rcaug \
    "++fraction=1.0" \
    "++batch_size=${BS}" \
    "++lr=${LR}" \
    "++epochs=20" \
    "++early_stop_patience=20" \
    "++wandb_mode=offline" \
    "++output_dir=outputs/bs_lr_grid_ag_k562/${TAG}"

END_TIME=$(date +%s)
echo "=== ${TAG} DONE at $(date) — wall time: $((END_TIME - START_TIME))s ==="
