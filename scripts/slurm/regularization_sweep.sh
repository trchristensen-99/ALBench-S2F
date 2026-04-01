#!/bin/bash
# Systematic regularization sweep for foundation S1 models.
# Maps the Pareto frontier between in-distribution and OOD performance.
#
# Parameters swept:
#   - Dropout: 0.1, 0.3, 0.5
#   - Weight decay: 1e-6, 1e-4, 1e-3
#   - Early stop patience: 3, 7, 15
#   = 27 configs per model
#
# Models: Enformer, Borzoi, NTv3 (K562 only, cached embeddings)
# Each config: 1 seed (42) for initial sweep, best configs get 3-seed followup
#
# Array 0-26: Enformer K562
# Array 27-53: Borzoi K562
# Array 54-80: NTv3 K562
#
# Uses V100 (cached embeddings, no encoder needed)
#
#SBATCH --job-name=reg_sweep
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=default
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

set -euo pipefail

set +u; source /etc/profile.d/modules.sh; set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
source scripts/slurm/setup_hpc_deps.sh

T=$SLURM_ARRAY_TASK_ID
echo "=== reg_sweep task=${T} node=${SLURMD_NODENAME} date=$(date) ==="

# Parameter grid
DROPOUTS=(0.1 0.3 0.5)
WEIGHT_DECAYS=(1e-6 1e-4 1e-3)
PATIENCES=(3 7 15)

# Determine model and config index
if [ $T -lt 27 ]; then
    MODEL="enformer"
    EMBED_DIM=3072
    CACHE_DIR="outputs/enformer_k562_cached/embedding_cache"
    CONFIG_IDX=$T
elif [ $T -lt 54 ]; then
    MODEL="borzoi"
    EMBED_DIM=1536
    CACHE_DIR="outputs/borzoi_k562_cached/embedding_cache"
    CONFIG_IDX=$((T - 27))
else
    MODEL="ntv3_post"
    EMBED_DIM=1536
    CACHE_DIR="outputs/ntv3_post_k562_cached/embedding_cache"
    CONFIG_IDX=$((T - 54))
fi

# Decode config index → dropout, wd, patience
DO_IDX=$((CONFIG_IDX / 9))
WD_IDX=$(( (CONFIG_IDX % 9) / 3 ))
PAT_IDX=$((CONFIG_IDX % 3))

DO="${DROPOUTS[$DO_IDX]}"
WD="${WEIGHT_DECAYS[$WD_IDX]}"
PAT="${PATIENCES[$PAT_IDX]}"

LABEL="do${DO}_wd${WD}_pat${PAT}"
OUT_DIR="outputs/regularization_sweep/${MODEL}/${LABEL}"

echo "Model: ${MODEL}, Config: ${LABEL}"
echo "  dropout=${DO}, weight_decay=${WD}, patience=${PAT}"

# Use lr=0.001 for all (standard for cached training)
uv run --no-sync python experiments/train_foundation_cached.py \
    ++model_name="${MODEL}" \
    ++cell_line=k562 \
    ++cache_dir="${CACHE_DIR}" \
    ++embed_dim="${EMBED_DIM}" \
    ++output_dir="${OUT_DIR}" \
    ++seed=42 \
    ++lr=0.001 \
    ++weight_decay="${WD}" \
    ++dropout="${DO}" \
    ++epochs=100 \
    ++early_stop_patience="${PAT}"

echo "=== Done: $(date) ==="
