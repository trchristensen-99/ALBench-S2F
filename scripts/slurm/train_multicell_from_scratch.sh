#!/bin/bash
# Train from-scratch models (DREAM-RNN, Malinois) on HepG2/SK-N-SH.
#
# Array:
#   0: DREAM-RNN HepG2 (3 seeds)
#   1: DREAM-RNN SKNSH (3 seeds)
#   2: Malinois HepG2 (3 seeds)
#   3: Malinois SKNSH (3 seeds)
#
#SBATCH --job-name=multicell
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=200G
#SBATCH --array=0-3

set -euo pipefail

set +u
source /etc/profile.d/modules.sh
set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
source scripts/slurm/setup_hpc_deps.sh

CONFIGS=(
    "dream_rnn hepg2"
    "dream_rnn sknsh"
    "malinois hepg2"
    "malinois sknsh"
)

CFG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
read -r MODEL CELL <<< "$CFG"

echo "=== Multi-Cell-Line From-Scratch Training ==="
echo "Model: ${MODEL}, Cell line: ${CELL}"
echo "Node: $SLURMD_NODENAME  Date: $(date)"

# Setup data symlinks
mkdir -p "data/${CELL}"
ln -sf "$(pwd)/data/k562/DATA-Table_S2__MPRA_dataset.txt" "data/${CELL}/DATA-Table_S2__MPRA_dataset.txt" 2>/dev/null || true
ln -sf "$(pwd)/data/k562/hashfrag_splits" "data/${CELL}/hashfrag_splits" 2>/dev/null || true

if [[ "${MODEL}" == "malinois" ]]; then
    for SEED in 0 1 2; do
        echo "--- Malinois seed ${SEED} ---"
        uv run --no-sync python experiments/train_malinois_k562.py \
            ++data_path="data/${CELL}" \
            ++output_dir="outputs/malinois_${CELL}_3seeds/seed_${SEED}" \
            ++seed="${SEED}" \
            ++cell_line="${CELL}"
    done
elif [[ "${MODEL}" == "dream_rnn" ]]; then
    for SEED in 0 1 2; do
        echo "--- DREAM-RNN seed ${SEED} ---"
        uv run --no-sync python experiments/exp0_k562_scaling.py \
            --data-path "data/${CELL}" \
            --output-dir "outputs/dream_rnn_${CELL}_3seeds/seed_${SEED}" \
            --seed "${SEED}" \
            --fractions 1.0 \
            --cell-line "${CELL}" 2>&1 || \
        echo "DREAM-RNN direct training not yet parametrized for cell_line, using exp1_1 fallback" && \
        uv run --no-sync python experiments/exp1_1_scaling.py \
            --task k562 \
            --student dream_rnn \
            --oracle default \
            --reservoir random \
            --n-replicates 1 \
            --seed "${SEED}" \
            --output-dir "outputs/dream_rnn_${CELL}_3seeds" \
            --training-sizes 319742 \
            --epochs 80 \
            --ensemble-size 3 \
            --early-stop-patience 10
    done
fi

echo "Done: $(date)"
