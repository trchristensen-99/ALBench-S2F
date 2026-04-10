#!/bin/bash
# Train LegNet on expanded dataset (Gosai + Agarwal negatives) vs baseline.
#
# Array tasks:
#   0-2: Expanded (3 seeds)
#   3-5: Baseline Gosai-only (3 seeds, for fair comparison)
#
# Submit:
#   /cm/shared/apps/slurm/current/bin/sbatch --array=0-5 scripts/slurm/legnet_expanded_experiment.sh
#
#SBATCH --job-name=lgnt_expand
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=default
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

set -euo pipefail
set +u; source /etc/profile.d/modules.sh; set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
source scripts/slurm/setup_hpc_deps.sh

T=$SLURM_ARRAY_TASK_ID
SEEDS=(42 142 242)

echo "=== LegNet expanded experiment: task=${T} $(date) ==="

# Step 0: Build expanded dataset (only once, first task)
if [ ! -f "data/k562_expanded/metadata.json" ]; then
    echo "Building expanded dataset..."
    uv run --no-sync python scripts/build_expanded_k562_dataset.py
fi

if [ "${T}" -lt 3 ]; then
    # ── Expanded training (tasks 0-2) ──
    SEED=${SEEDS[$T]}
    OUT="outputs/legnet_expanded/seed_${SEED}"
    echo "Expanded training: seed=${SEED}"

    uv run --no-sync python scripts/train_legnet_expanded.py \
        --seed "${SEED}" \
        --output-dir "${OUT}" \
        --lr 0.001 --batch-size 512 \
        --epochs 80 --patience 10
else
    # ── Baseline Gosai-only (tasks 3-5) ──
    IDX=$((T - 3))
    SEED=${SEEDS[$IDX]}
    OUT="outputs/legnet_baseline_comparison/seed_${SEED}"
    echo "Baseline (Gosai-only) training: seed=${SEED}"

    # Train on Gosai-only but evaluate on same test sets
    uv run --no-sync python scripts/train_legnet_expanded.py \
        --seed "${SEED}" \
        --output-dir "${OUT}" \
        --expanded-path "data/k562_expanded_empty" \
        --lr 0.001 --batch-size 512 \
        --epochs 80 --patience 10
fi

echo "=== Done: task=${T} $(date) ==="
