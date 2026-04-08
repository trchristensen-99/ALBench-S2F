#!/bin/bash
# Train S1 heads for all foundation models on HepG2/SK-N-SH.
# Reuses existing embedding caches (same sequences, different labels).
#
# Array: 0-5 for 3 models × 2 cell lines
#   0: Enformer HepG2
#   1: Enformer SKNSH
#   2: Borzoi HepG2
#   3: Borzoi SKNSH
#   4: NTv3-post HepG2
#   5: NTv3-post SKNSH
#
# Each runs 3 seeds internally.
#
#SBATCH --job-name=s1_multi
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=default
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --array=0-5

set -euo pipefail

set +u
source /etc/profile.d/modules.sh
set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
source scripts/slurm/setup_hpc_deps.sh

CONFIGS=(
    "enformer hepg2 outputs/enformer_k562_cached/embedding_cache 3072"
    "enformer sknsh outputs/enformer_k562_cached/embedding_cache 3072"
    "borzoi hepg2 outputs/borzoi_k562_cached/embedding_cache 1536"
    "borzoi sknsh outputs/borzoi_k562_cached/embedding_cache 1536"
    "ntv3_post hepg2 outputs/ntv3_post_k562_cached/embedding_cache 1536"
    "ntv3_post sknsh outputs/ntv3_post_k562_cached/embedding_cache 1536"
)

CFG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
read -r MODEL CELL CACHE_DIR EMBED_DIM <<< "$CFG"

echo "=== S1 Head Training ==="
echo "Model: ${MODEL}, Cell line: ${CELL}, Embed dim: ${EMBED_DIM}"
echo "Cache: ${CACHE_DIR}"
echo "Node: $SLURMD_NODENAME  Date: $(date)"

# Data path points to k562 (same raw file, hashfrag splits, and test sets for all cell lines)
# But test_sets are in data/{cell_line}/test_sets/
# We need to symlink so the data loader can find the raw data
mkdir -p "data/${CELL}"
ln -sf "$(pwd)/data/k562/DATA-Table_S2__MPRA_dataset.txt" "data/${CELL}/DATA-Table_S2__MPRA_dataset.txt" 2>/dev/null || true
ln -sf "$(pwd)/data/k562/hashfrag_splits" "data/${CELL}/hashfrag_splits" 2>/dev/null || true

OUT_DIR="outputs/${MODEL}_${CELL}_cached"

for SEED_IDX in 0 1 2; do
    echo "--- Seed ${SEED_IDX} ---"
    uv run --no-sync python experiments/train_foundation_cached.py \
        ++model_name="${MODEL}" \
        ++cache_dir="${CACHE_DIR}" \
        ++embed_dim="${EMBED_DIM}" \
        ++output_dir="${OUT_DIR}/seed_${SEED_IDX}" \
        ++data_path="data/${CELL}" \
        ++cell_line="${CELL}" \
        ++seed="${SEED_IDX}"
done

echo "Done: $(date)"
