#!/bin/bash
# AG fold-1 and all-folds multi-cell 3-seed training (seeds 1,2; seed 0 already done).
#
# Array mapping:
#   0-1:   AG fold-1 S1 HepG2  seeds 1,2
#   2-3:   AG fold-1 S1 SKNSH  seeds 1,2
#   4-5:   AG fold-1 S2 HepG2  seeds 1,2 (enc_lr=1e-4)
#   6-7:   AG fold-1 S2 SKNSH  seeds 1,2 (enc_lr=1e-4)
#   8-9:   AG all-folds S2 HepG2  seeds 1,2 (s2c: enc_lr=1e-4, head_lr=1e-3)
#   10-11: AG all-folds S2 SKNSH  seeds 1,2 (s2c: enc_lr=1e-4, head_lr=1e-3)
#
# Usage:
#   /cm/shared/apps/slurm/current/bin/sbatch --array=0-11 scripts/slurm/ag_multicell_3seeds.sh
#
#SBATCH --job-name=ag_mc_3s
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=default
#SBATCH --time=12:00:00
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
export PYTHONUNBUFFERED=1

FOLD1_WEIGHTS="/grid/wsbs/home_norepl/christen/alphagenome_weights/alphagenome-jax-fold_1"

# Config arrays (12 tasks)
#              0       1       2       3       4       5       6       7       8       9       10      11
MODELS=(  "fold_1" "fold_1" "fold_1" "fold_1" "fold_1" "fold_1" "fold_1" "fold_1" "all_folds" "all_folds" "all_folds" "all_folds")
STAGES=(  "s1"     "s1"     "s1"     "s1"     "s2"     "s2"     "s2"     "s2"     "s2"        "s2"        "s2"        "s2")
CELLS=(   "hepg2"  "hepg2"  "sknsh"  "sknsh"  "hepg2"  "hepg2"  "sknsh"  "sknsh"  "hepg2"    "hepg2"    "sknsh"     "sknsh")
SEEDS=(   1        2        1        2        1        2        1        2        1           2           1            2)

IDX=${SLURM_ARRAY_TASK_ID}
MODEL="${MODELS[$IDX]}"
STAGE="${STAGES[$IDX]}"
CELL="${CELLS[$IDX]}"
SEED="${SEEDS[$IDX]}"

echo "=== AG ${MODEL} ${STAGE} ${CELL} seed=${SEED} (task ${IDX}) ==="
echo "Node: $SLURMD_NODENAME  Date: $(date)"

# Setup data symlinks
mkdir -p "data/${CELL}"
ln -sf "$(pwd)/data/k562/DATA-Table_S2__MPRA_dataset.txt" "data/${CELL}/DATA-Table_S2__MPRA_dataset.txt" 2>/dev/null || true
ln -sf "$(pwd)/data/k562/hashfrag_splits" "data/${CELL}/hashfrag_splits" 2>/dev/null || true

# Ensure test sets exist
if [[ ! -f "data/${CELL}/test_sets/test_in_distribution_hashfrag.tsv" ]]; then
    uv run --no-sync python scripts/create_cellline_test_sets.py --cell-line "${CELL}"
fi

# Set weights path for fold-1 tasks
if [[ "${MODEL}" == "fold_1" ]]; then
    export ALPHAGENOME_WEIGHTS="${FOLD1_WEIGHTS}"
fi
# For all_folds, don't set ALPHAGENOME_WEIGHTS (use default all_folds)

if [[ "${STAGE}" == "s1" ]]; then
    OUT_DIR="outputs/ag_fold_1_${CELL}_s1/seed_${SEED}"

    # Skip if already complete
    if [[ -f "${OUT_DIR}/size_319742/run_0/result.json" ]]; then
        echo "SKIP: result already exists at ${OUT_DIR}"
        exit 0
    fi

    uv run --no-sync python experiments/exp1_1_scaling.py \
        --task k562 \
        --student alphagenome_k562_s1 \
        --oracle ground_truth \
        --cell-line "${CELL}" \
        --reservoir genomic \
        --n-replicates 1 \
        --no-hp-sweep \
        --seed "${SEED}" \
        --output-dir "${OUT_DIR}" \
        --training-sizes 319742 \
        --epochs 50 \
        --early-stop-patience 7

elif [[ "${STAGE}" == "s2" ]]; then
    if [[ "${MODEL}" == "fold_1" ]]; then
        OUT_DIR="outputs/ag_fold_1_${CELL}_s2/seed_${SEED}"
    else
        OUT_DIR="outputs/ag_all_folds_${CELL}_s2_v2/seed_${SEED}"
    fi

    # Skip if already complete
    if [[ -f "${OUT_DIR}/size_319742/run_0/result.json" ]]; then
        echo "SKIP: result already exists at ${OUT_DIR}"
        exit 0
    fi

    uv run --no-sync python experiments/exp1_1_scaling.py \
        --task k562 \
        --student alphagenome_k562_s2 \
        --oracle ground_truth \
        --cell-line "${CELL}" \
        --reservoir genomic \
        --n-replicates 1 \
        --no-hp-sweep \
        --seed "${SEED}" \
        --output-dir "${OUT_DIR}" \
        --training-sizes 319742 \
        --epochs 50 \
        --early-stop-patience 10
fi

echo "Done: $(date)"
