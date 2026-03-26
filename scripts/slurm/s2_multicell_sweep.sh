#!/bin/bash
# Stage 2 HP sweep for models where S2 < S1 on HepG2/SK-N-SH.
#
# Tests multiple encoder_lr values to find the optimal S2 config per cell.
# Uses the K562 S2 config as the starting point (enc_lr=1e-4, head_lr=1e-3).
#
# Array mapping:
#   0: AG all-folds HepG2 (s2c config: enc_lr=1e-4, head_lr=1e-3)
#   1: AG all-folds SKNSH (s2c config)
#   2: AG fold-1 HepG2 (enc_lr=1e-4)
#   3: AG fold-1 SKNSH (enc_lr=1e-4)
#   4: AG all-folds HepG2 (enc_lr=5e-5)
#   5: AG all-folds SKNSH (enc_lr=5e-5)
#   6: AG fold-1 HepG2 (enc_lr=5e-5)
#   7: AG fold-1 SKNSH (enc_lr=5e-5)
#
# Usage:
#   sbatch --array=0-7 scripts/slurm/s2_multicell_sweep.sh
#
#SBATCH --job-name=s2_sweep
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

# Config arrays
MODELS=("all_folds" "all_folds" "fold_1" "fold_1" "all_folds" "all_folds" "fold_1" "fold_1")
CELLS=("hepg2" "sknsh" "hepg2" "sknsh" "hepg2" "sknsh" "hepg2" "sknsh")
ENC_LRS=("1e-4" "1e-4" "1e-4" "1e-4" "5e-5" "5e-5" "5e-5" "5e-5")

MODEL="${MODELS[$SLURM_ARRAY_TASK_ID]}"
CELL="${CELLS[$SLURM_ARRAY_TASK_ID]}"
ENC_LR="${ENC_LRS[$SLURM_ARRAY_TASK_ID]}"

echo "=== S2 Sweep: ${MODEL} ${CELL} enc_lr=${ENC_LR} ==="
echo "Node: $SLURMD_NODENAME  Date: $(date)"

# Setup data symlinks
mkdir -p "data/${CELL}"
ln -sf "$(pwd)/data/k562/DATA-Table_S2__MPRA_dataset.txt" "data/${CELL}/DATA-Table_S2__MPRA_dataset.txt" 2>/dev/null || true
ln -sf "$(pwd)/data/k562/hashfrag_splits" "data/${CELL}/hashfrag_splits" 2>/dev/null || true

# Set weights path
if [[ "${MODEL}" == "fold_1" ]]; then
    export ALPHAGENOME_WEIGHTS="/grid/wsbs/home_norepl/christen/alphagenome_weights/alphagenome-jax-fold_1"
    OUT_DIR="outputs/ag_fold_1_${CELL}_s2_sweep/enc_lr_${ENC_LR}"
    STUDENT="alphagenome_k562_s2"
else
    export ALPHAGENOME_WEIGHTS="/grid/wsbs/home_norepl/christen/alphagenome_weights/alphagenome-jax-all_folds-v1"
    OUT_DIR="outputs/ag_all_folds_${CELL}_s2_sweep/enc_lr_${ENC_LR}"
    STUDENT="alphagenome_k562_s2"
fi

# Train with ground_truth labels, proper S2 config
# The S2_CONFIG in exp1_1_scaling.py uses head_lr=1e-3, unfreeze blocks 4,5
# We override the encoder LR via the HP grid
uv run --no-sync python experiments/exp1_1_scaling.py \
    --task k562 \
    --student "${STUDENT}" \
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

echo "Done: $(date)"
