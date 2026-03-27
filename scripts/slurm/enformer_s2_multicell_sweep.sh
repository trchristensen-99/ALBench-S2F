#!/bin/bash
# Enformer S2 sweep for HepG2/SK-N-SH with different configs.
#
# The original S2 used unfreeze_mode=transformer, which may not be optimal.
# K562 used unfreeze_mode=all and got a big improvement. Try both.
#
# Array:
#   0: Enformer S2 HepG2 unfreeze=all
#   1: Enformer S2 SKNSH unfreeze=all
#   2: Enformer S2 HepG2 unfreeze=last4
#   3: Enformer S2 SKNSH unfreeze=last4
#
# Usage:
#   sbatch --array=0-3 scripts/slurm/enformer_s2_multicell_sweep.sh
#
#SBATCH --job-name=enf_s2_sweep
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

CELLS=("hepg2" "sknsh" "hepg2" "sknsh")
UNFREEZE=("all" "all" "last4" "last4")

CELL="${CELLS[$SLURM_ARRAY_TASK_ID]}"
UF="${UNFREEZE[$SLURM_ARRAY_TASK_ID]}"

echo "=== Enformer S2 Sweep: ${CELL} unfreeze=${UF} ==="
echo "Node: $SLURMD_NODENAME  Date: $(date)"

# Setup data
mkdir -p "data/${CELL}"
ln -sf "$(pwd)/data/k562/DATA-Table_S2__MPRA_dataset.txt" "data/${CELL}/DATA-Table_S2__MPRA_dataset.txt" 2>/dev/null || true
ln -sf "$(pwd)/data/k562/hashfrag_splits" "data/${CELL}/hashfrag_splits" 2>/dev/null || true

# Use best S1 seed as starting point
S1_DIR="outputs/enformer_${CELL}_cached/seed_0/seed_0"
OUT_DIR="outputs/enformer_${CELL}_s2_sweep/uf_${UF}"

uv run --no-sync python experiments/train_foundation_stage2.py \
    ++model_name=enformer \
    ++stage1_result_dir="${S1_DIR}" \
    ++output_dir="${OUT_DIR}" \
    ++data_path="data/${CELL}" \
    ++cell_line="${CELL}" \
    ++seed=42 \
    ++epochs=15 \
    ++batch_size=4 \
    ++grad_accum_steps=2 \
    ++head_lr=0.001 \
    ++encoder_lr=0.0001 \
    ++weight_decay=1e-6 \
    ++hidden_dim=512 \
    ++dropout=0.1 \
    ++early_stop_patience=5 \
    ++max_train_sequences=20000 \
    ++max_val_sequences=2000 \
    ++rc_aug=True \
    ++unfreeze_mode="${UF}" \
    ++grad_clip=1.0 \
    ++amp_mode=bfloat16

echo "Done: $(date)"
