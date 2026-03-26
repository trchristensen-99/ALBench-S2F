#!/bin/bash
# Train AG fold-1 S1 + S2 for HepG2 and SK-N-SH.
#
# S1: Head-only training using fold_1 encoder (full encoder pass, not cached).
# S2: Fine-tune fold_1 encoder + head on cell-line-specific labels.
#
# Array:
#   0: fold-1 S1 HepG2
#   1: fold-1 S1 SKNSH
#   2: fold-1 S2 HepG2
#   3: fold-1 S2 SKNSH
#
# Usage:
#   sbatch --array=0-3 scripts/slurm/train_ag_fold1_multicell.sh
#
#SBATCH --job-name=ag_fold1_mc
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=200G

set -euo pipefail

set +u
source /etc/profile.d/modules.sh
set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
source scripts/slurm/setup_hpc_deps.sh

export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_command_buffer="

FOLD1_WEIGHTS="/grid/wsbs/home_norepl/christen/alphagenome_weights/alphagenome-jax-fold_1"

STAGES=("s1" "s1" "s2" "s2")
CELLS=("hepg2" "sknsh" "hepg2" "sknsh")
STAGE="${STAGES[$SLURM_ARRAY_TASK_ID]}"
CELL="${CELLS[$SLURM_ARRAY_TASK_ID]}"

echo "=== AG fold-1 ${STAGE} ${CELL} ==="
echo "Node: $SLURMD_NODENAME  Date: $(date)"

# Setup data symlinks
mkdir -p "data/${CELL}"
ln -sf "$(pwd)/data/k562/DATA-Table_S2__MPRA_dataset.txt" "data/${CELL}/DATA-Table_S2__MPRA_dataset.txt" 2>/dev/null || true
ln -sf "$(pwd)/data/k562/hashfrag_splits" "data/${CELL}/hashfrag_splits" 2>/dev/null || true

# Ensure test sets exist
if [[ ! -f "data/${CELL}/test_sets/test_in_distribution_hashfrag.tsv" ]]; then
    uv run --no-sync python scripts/create_cellline_test_sets.py --cell-line "${CELL}"
fi

if [[ "${STAGE}" == "s1" ]]; then
    # S1: Use exp1_1_scaling.py with ground_truth oracle + fold_1 encoder
    # We use alphagenome_k562_s1 student type but override the weights path
    OUT_DIR="outputs/ag_fold_1_${CELL}_s1"
    export ALPHAGENOME_WEIGHTS="${FOLD1_WEIGHTS}"

    uv run --no-sync python experiments/exp1_1_scaling.py \
        --task k562 \
        --student alphagenome_k562_s1 \
        --oracle ground_truth \
        --cell-line "${CELL}" \
        --reservoir genomic \
        --n-replicates 1 \
        --no-hp-sweep \
        --seed 42 \
        --output-dir "${OUT_DIR}" \
        --training-sizes 319742 \
        --epochs 50 \
        --early-stop-patience 7

elif [[ "${STAGE}" == "s2" ]]; then
    # S2: Fine-tune fold_1 encoder
    OUT_DIR="outputs/ag_fold_1_${CELL}_s2"
    export ALPHAGENOME_WEIGHTS="${FOLD1_WEIGHTS}"

    uv run --no-sync python experiments/exp1_1_scaling.py \
        --task k562 \
        --student alphagenome_k562_s2 \
        --oracle ground_truth \
        --cell-line "${CELL}" \
        --reservoir genomic \
        --n-replicates 1 \
        --no-hp-sweep \
        --seed 42 \
        --output-dir "${OUT_DIR}" \
        --training-sizes 319742 \
        --epochs 50 \
        --early-stop-patience 10
fi

echo "Done: $(date)"
