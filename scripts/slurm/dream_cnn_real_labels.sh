#!/bin/bash
# DREAM-CNN with real labels on all 3 cell lines (for Exp0 comparison baseline).
# Also serves as the bar-plot single-model DREAM-CNN result.
#
# Array:
#   0-2: K562 seeds 0,1,2
#   3-5: HepG2 seeds 0,1,2
#   6-8: SKNSH seeds 0,1,2
#
# Usage:
#   sbatch --array=0-8 scripts/slurm/dream_cnn_real_labels.sh
#
#SBATCH --job-name=dcnn_real
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

CELLS=("k562" "k562" "k562" "hepg2" "hepg2" "hepg2" "sknsh" "sknsh" "sknsh")
SEEDS=(0 1 2 0 1 2 0 1 2)

CELL="${CELLS[$SLURM_ARRAY_TASK_ID]}"
SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"

echo "=== DREAM-CNN real labels ==="
echo "Cell: ${CELL}, Seed: ${SEED}"
echo "Node: $SLURMD_NODENAME  Date: $(date)"

# Setup data symlinks
if [[ "${CELL}" != "k562" ]]; then
    mkdir -p "data/${CELL}"
    ln -sf "$(pwd)/data/k562/DATA-Table_S2__MPRA_dataset.txt" "data/${CELL}/DATA-Table_S2__MPRA_dataset.txt" 2>/dev/null || true
    ln -sf "$(pwd)/data/k562/hashfrag_splits" "data/${CELL}/hashfrag_splits" 2>/dev/null || true
fi

CMD=(
    uv run --no-sync python experiments/exp1_1_scaling.py
    --task k562
    --student dream_cnn
    --oracle ground_truth
    --reservoir genomic
    --n-replicates 1
    --no-hp-sweep
    --seed "${SEED}"
    --output-dir "outputs/dream_cnn_${CELL}_real/seed_${SEED}"
    --training-sizes 319742
    --epochs 80
    --ensemble-size 1
    --early-stop-patience 10
)

[[ "${CELL}" != "k562" ]] && CMD+=(--cell-line "${CELL}")

"${CMD[@]}"

echo "Done: $(date)"
