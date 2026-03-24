#!/bin/bash
# Train from-scratch models using chromosome-based splits (chr7,13 test).
#
# Array:
#   0: DREAM-RNN K562 (3 seeds)
#   1: Malinois K562 (3 seeds)
#   2: DREAM-CNN K562 (3 seeds)
#   3: DREAM-RNN HepG2 (3 seeds)
#   4: Malinois HepG2 (3 seeds)
#   5: DREAM-RNN SKNSH (3 seeds)
#   6: Malinois SKNSH (3 seeds)
#
#SBATCH --job-name=chr_split
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=200G
#SBATCH --array=0-6

set -euo pipefail

set +u
source /etc/profile.d/modules.sh
set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
source scripts/slurm/setup_hpc_deps.sh

CONFIGS=(
    "dream_rnn k562"
    "malinois k562"
    "dream_cnn k562"
    "dream_rnn hepg2"
    "malinois hepg2"
    "dream_rnn sknsh"
    "malinois sknsh"
)

CFG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
read -r MODEL CELL <<< "$CFG"

echo "=== Chromosome Split Training ==="
echo "Model: ${MODEL}, Cell line: ${CELL}"
echo "Test: chr7+13, Val: chr19+21+X"
echo "Node: $SLURMD_NODENAME  Date: $(date)"

# Setup data symlinks for non-K562 cell lines
if [[ "${CELL}" != "k562" ]]; then
    mkdir -p "data/${CELL}"
    ln -sf "$(pwd)/data/k562/DATA-Table_S2__MPRA_dataset.txt" "data/${CELL}/DATA-Table_S2__MPRA_dataset.txt" 2>/dev/null || true
fi

CELL_LINE_MAP_K562="K562_log2FC"
CELL_LINE_MAP_HEPG2="HepG2_log2FC"
CELL_LINE_MAP_SKNSH="SKNSH_log2FC"

case "${CELL}" in
    k562) LABEL_COL="${CELL_LINE_MAP_K562}" ;;
    hepg2) LABEL_COL="${CELL_LINE_MAP_HEPG2}" ;;
    sknsh) LABEL_COL="${CELL_LINE_MAP_SKNSH}" ;;
esac

OUT_DIR="outputs/chr_split/${CELL}/${MODEL}"

if [[ "${MODEL}" == "malinois" ]]; then
    for SEED in 0 1 2; do
        echo "--- ${MODEL} seed ${SEED} ---"
        uv run --no-sync python experiments/train_malinois_k562.py \
            ++data_path="data/${CELL}" \
            ++output_dir="${OUT_DIR}/seed_${SEED}" \
            ++seed="${SEED}" \
            ++cell_line="${CELL}"
    done
elif [[ "${MODEL}" == "dream_rnn" ]] || [[ "${MODEL}" == "dream_cnn" ]]; then
    # Use exp1_1_scaling with full dataset size and random reservoir
    uv run --no-sync python experiments/exp1_1_scaling.py \
        --task k562 \
        --student "${MODEL}" \
        --oracle default \
        --reservoir random \
        --n-replicates 3 \
        --seed 42 \
        --output-dir "${OUT_DIR}" \
        --training-sizes 319742 \
        --epochs 80 \
        --ensemble-size 3 \
        --early-stop-patience 10
fi

echo "Done: $(date)"
