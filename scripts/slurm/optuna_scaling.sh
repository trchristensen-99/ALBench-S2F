#!/bin/bash
# Lightweight Optuna HP search (no Ray overhead).
#
# 6 strategies × 6 sizes = 36 jobs per seed.
# Each runs 10 Optuna trials + 3 replicates.
#
# Array: strat_idx * 6 + size_idx (0-35)
#
#SBATCH --job-name=opt_scl
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
#SBATCH --partition=gpuq
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

set -euo pipefail
set +u; source /etc/profile.d/modules.sh; set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD"
export TORCHDYNAMO_DISABLE=1
source scripts/slurm/setup_hpc_deps.sh

T=$SLURM_ARRAY_TASK_ID

STRATS=("random" "genomic" "prm_1pct" "prm_20pct" "motif_grammar" "evoaug_heavy")
SIZES=(1000 2000 5000 10000 20000 50000)

STRAT_IDX=$((T / 6))
SIZE_IDX=$((T % 6))
STRAT=${STRATS[$STRAT_IDX]}
SIZE=${SIZES[$SIZE_IDX]}
SEED=${OPTUNA_SEED:-42}

echo "=== Optuna: ${STRAT} n=${SIZE} seed=${SEED} — $(date) ==="

uv run --no-sync python scripts/optuna_legnet_scaling.py \
    --strategy "${STRAT}" \
    --size "${SIZE}" \
    --seed "${SEED}" \
    --n-trials 10

echo "=== DONE — $(date) ==="
