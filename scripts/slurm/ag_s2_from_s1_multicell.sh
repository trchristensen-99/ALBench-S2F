#!/bin/bash
# AG S2 fine-tuning for HepG2/SK-N-SH, initialized from S1 head checkpoints.
#
# The previous S2 runs used exp1_1_scaling.py which reinitializes the head
# from random weights. The K562 S2 used train_stage2_k562_hashfrag.py which
# loads the trained S1 head. This script does the same for HepG2/SK-N-SH.
#
# Array:
#   0: AG all-folds S2 HepG2 (init from S1 seed 0)
#   1: AG all-folds S2 SKNSH (init from S1 seed 0)
#   2: AG fold-1 S2 HepG2 (init from S1 seed 42)
#   3: AG fold-1 S2 SKNSH (init from S1 seed 42)
#
# Usage:
#   sbatch --array=0-3 scripts/slurm/ag_s2_from_s1_multicell.sh
#
#SBATCH --job-name=ag_s2_s1init
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=default
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

T=$SLURM_ARRAY_TASK_ID

MODELS=("all_folds" "all_folds" "fold_1" "fold_1")
CELLS=("hepg2" "sknsh" "hepg2" "sknsh")
S1_DIRS=(
    "outputs/ag_hashfrag_hepg2_cached/seed_0"
    "outputs/ag_hashfrag_sknsh_cached/seed_0"
    "outputs/ag_fold_1_hepg2_s1/genomic/n319742/hp0/seed42"
    "outputs/ag_fold_1_sknsh_s1/genomic/n319742/hp0/seed42"
)

MODEL="${MODELS[$T]}"
CELL="${CELLS[$T]}"
S1_DIR="${S1_DIRS[$T]}"
OUT_DIR="outputs/ag_${MODEL}_${CELL}_s2_from_s1"

echo "=== AG ${MODEL} S2 from S1 — ${CELL} ==="
echo "S1 dir: ${S1_DIR}"
echo "Output: ${OUT_DIR}"
echo "Node: $SLURMD_NODENAME  Date: $(date)"

# Set weights for fold-1
if [[ "${MODEL}" == "fold_1" ]]; then
    export ALPHAGENOME_WEIGHTS="/grid/wsbs/home_norepl/christen/alphagenome_weights/alphagenome-jax-fold_1"
fi

# Use the SAME script as K562 S2 but with cell-line override
uv run --no-sync python experiments/train_stage2_k562_hashfrag.py \
    ++stage1_dir="${S1_DIR}" \
    ++output_dir="${OUT_DIR}" \
    ++data_path="data/k562" \
    ++cell_line="${CELL}" \
    ++seed=42 \
    ++encoder_lr=0.0001 \
    ++head_lr=0.001 \
    ++weight_decay=1e-6 \
    ++epochs=30 \
    ++early_stop_patience=7 \
    ++warmup_epochs=3 \
    ++batch_size=128

echo "Done: $(date)"
