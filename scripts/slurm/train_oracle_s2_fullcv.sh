#!/bin/bash
# DEFAULT oracle — Stage-2 encoder fine-tune, full-dataset random 10-fold CV.
#
# Each fold initialises from the matching Stage-1 fold checkpoint
# (outputs/oracle_full856k_clean/s1/oracle_${FOLD}) and fine-tunes the top
# encoder downres blocks (proven s2c: enc_lr=1e-4, head_lr=1e-3, unfreeze 4,5)
# on the FULL 856,252-row dataset under the identical seed=42 10-fold split.
#
# Submit only after all 10 S1 folds are present, e.g.:
#   /cm/shared/apps/slurm/current/bin/sbatch --array=0-3 --qos=default --time=12:00:00 \
#       scripts/slurm/train_oracle_s2_fullcv.sh
#   /cm/shared/apps/slurm/current/bin/sbatch --array=4-9 --qos=slow_nice --time=12:00:00 \
#       scripts/slurm/train_oracle_s2_fullcv.sh
#
#SBATCH --job-name=orc_s2_fullcv
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=200G

set -euo pipefail

set +u; source /etc/profile.d/modules.sh; set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
source scripts/slurm/setup_hpc_deps.sh
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_command_buffer="

FOLD=$SLURM_ARRAY_TASK_ID
CACHE_DIR="outputs/oracle_full856k_clean/embedding_cache"
S1_DIR="outputs/oracle_full856k_clean/s1/oracle_${FOLD}"
OUT_DIR="outputs/oracle_full856k_clean/s2/fold_${FOLD}"

echo "=== S2 full-CV oracle: fold ${FOLD} node=${SLURMD_NODENAME} $(date) ==="

# Safety: this fold's S1 checkpoint must exist.
if [ ! -d "${S1_DIR}/best_model/checkpoint" ]; then
    echo "ERROR: Stage-1 checkpoint missing at ${S1_DIR}/best_model/checkpoint — aborting."
    exit 2
fi

uv run --no-sync python experiments/train_oracle_s2_fullcv.py \
    --cache-dir "${CACHE_DIR}" \
    --stage1-dir "${S1_DIR}" \
    --output-dir "${OUT_DIR}" \
    --fold-id "${FOLD}" \
    --n-folds 10 \
    --epochs 15 \
    --early-stop-patience 4

echo "=== Done: fold ${FOLD} $(date) ==="
