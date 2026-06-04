#!/bin/bash
# DEFAULT oracle — Stage-1 head, full-dataset random 10-fold CV.
#
# Trains the BodaFlatten head on the frozen AlphaGenome encoder using the
# pre-built clean full embedding cache (ref + alt + OOD designed, 856,252 rows).
# Each fold is a deterministic random 90/10 split (seed=42 permutation); the
# 10-fold ensemble collectively covers the entire dataset. INCLUDES the 22,962
# OOD designed high-activity sequences (this is "the standard" default oracle).
#
# Head-only training (frozen encoder, detach_backbone) → val Pearson ~0.90.
# Each fold ~15-20 min on H100; fast qos runs 2 at a time (~5 waves, well under 4h).
#
# Submit:
#   /cm/shared/apps/slurm/current/bin/sbatch --array=0-9 scripts/slurm/train_oracle_s1_fullcv.sh
#
#SBATCH --job-name=orc_s1_fullcv
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=fast
#SBATCH --time=04:00:00
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
OUT_DIR="outputs/oracle_full856k_clean/s1/oracle_${FOLD}"

echo "=== S1 full-CV oracle: fold ${FOLD} node=${SLURMD_NODENAME} $(date) ==="

# Safety: the clean cache must already exist (built in prior session).
if [ ! -f "${CACHE_DIR}/train_canonical.npy" ] || [ ! -f "${CACHE_DIR}/all_labels.npy" ]; then
    echo "ERROR: clean embedding cache missing at ${CACHE_DIR} — aborting."
    echo "Expected train_canonical.npy + train_rc.npy + all_labels.npy"
    exit 2
fi

uv run --no-sync python experiments/train_oracle_s1_fullcv.py \
    --cache-dir "${CACHE_DIR}" \
    --output-dir "${OUT_DIR}" \
    --fold-id "${FOLD}" \
    --n-folds 10 \
    --epochs 20 \
    --early-stop-patience 5

echo "=== Done: fold ${FOLD} $(date) ==="
