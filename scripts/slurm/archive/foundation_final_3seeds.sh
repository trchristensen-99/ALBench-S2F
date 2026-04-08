#!/bin/bash
# Final 3-seed evaluation for foundation models with grid-search-optimized hyperparameters.
# Run AFTER foundation_grid_search.sh completes and you've identified the best configs.
#
# UPDATE the *_LR, *_WD, *_DO variables below with the best values from the grid search.
#
# Array: 0-2 = NTv3 seeds, 3-5 = Borzoi seeds, 6-8 = Enformer seeds
#
# Submit:
#   /cm/shared/apps/slurm/current/bin/sbatch scripts/slurm/foundation_final_3seeds.sh
#
#SBATCH --job-name=fm_final
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-8

set -euo pipefail

set +u
source /etc/profile.d/modules.sh
set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
source scripts/slurm/setup_hpc_deps.sh

# ── Best hyperparameters from grid search (UPDATE THESE) ─────────────────────
# NTv3
NTV3_LR=0.001
NTV3_WD=0.000001
NTV3_DO=0.1

# Borzoi
BORZOI_LR=0.001
BORZOI_WD=0.000001
BORZOI_DO=0.1

# Enformer
ENFORMER_LR=0.001
ENFORMER_WD=0.000001
ENFORMER_DO=0.1

# ── Map array index to model ─────────────────────────────────────────────────
IDX=${SLURM_ARRAY_TASK_ID}

if [ "$IDX" -lt 3 ]; then
    MODEL=ntv3
    CACHE_DIR=outputs/ntv3_k562_cached/embedding_cache
    EMBED_DIM=1536
    OUT_DIR=outputs/ntv3_k562_cached
    LR=$NTV3_LR; WD=$NTV3_WD; DO=$NTV3_DO
elif [ "$IDX" -lt 6 ]; then
    MODEL=borzoi
    CACHE_DIR=outputs/borzoi_k562_cached/embedding_cache
    EMBED_DIM=1536
    OUT_DIR=outputs/borzoi_k562_cached
    LR=$BORZOI_LR; WD=$BORZOI_WD; DO=$BORZOI_DO
else
    MODEL=enformer
    CACHE_DIR=outputs/enformer_k562_cached/embedding_cache
    EMBED_DIM=3072
    OUT_DIR=outputs/enformer_k562_cached
    LR=$ENFORMER_LR; WD=$ENFORMER_WD; DO=$ENFORMER_DO
fi

echo "Final eval: ${MODEL} (task ${IDX}) lr=${LR} wd=${WD} do=${DO}"
echo "Node: ${SLURMD_NODENAME}  Date: $(date)"

uv run --no-sync python experiments/train_foundation_cached.py \
    ++model_name="${MODEL}" \
    ++cache_dir="${CACHE_DIR}" \
    ++embed_dim="${EMBED_DIM}" \
    ++output_dir="${OUT_DIR}" \
    ++lr="${LR}" \
    ++weight_decay="${WD}" \
    ++dropout="${DO}"

echo "${MODEL} seed DONE — $(date)"
