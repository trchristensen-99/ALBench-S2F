#!/bin/bash
# Train AG S1 heads (cached embeddings) for K562, HepG2, SK-N-SH.
# Uses the existing AG hashfrag embedding cache with cell-line-specific labels.
#
# Array:
#   0: K562 (3 seeds)
#   1: HepG2 (3 seeds)
#   2: SKNSH (3 seeds)
#
#SBATCH --job-name=ag_s1_multi
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --array=0-2

set -euo pipefail

set +u
source /etc/profile.d/modules.sh
set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
source scripts/slurm/setup_hpc_deps.sh

CELLS=("k562" "hepg2" "sknsh")
CELL="${CELLS[$SLURM_ARRAY_TASK_ID]}"

echo "=== AG S1 Head Training ==="
echo "Cell line: ${CELL}"
echo "Node: $SLURMD_NODENAME  Date: $(date)"

# Symlink data for non-K562
if [[ "${CELL}" != "k562" ]]; then
    mkdir -p "data/${CELL}"
    ln -sf "$(pwd)/data/k562/DATA-Table_S2__MPRA_dataset.txt" "data/${CELL}/DATA-Table_S2__MPRA_dataset.txt" 2>/dev/null || true
    ln -sf "$(pwd)/data/k562/hashfrag_splits" "data/${CELL}/hashfrag_splits" 2>/dev/null || true
fi

OUT_DIR="outputs/ag_hashfrag_${CELL}_cached"

for SEED in 0 1 2; do
    echo "--- ${CELL} seed ${SEED} ---"
    uv run --no-sync python experiments/train_oracle_alphagenome_hashfrag_cached.py \
        ++output_dir="${OUT_DIR}/seed_${SEED}" \
        ++data_path="data/${CELL}" \
        ++cell_line="${CELL}" \
        ++seed="${SEED}"
done

echo "Done: $(date)"
