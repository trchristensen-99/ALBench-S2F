#!/bin/bash
# Optuna HP search for 1M+ sizes (extends scaling curves).
#
# 5 strategies × 2 sizes (1M, 2M) = 10 jobs per seed.
# (5M needs pool > 2M — skip for now)
# Uses labeled_pools_2m which has 2M sequences.
#
# Array: strat_idx * 2 + size_idx (0-9)
#
#SBATCH --job-name=opt_1m
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
#SBATCH --partition=gpuq
#SBATCH --time=12:00:00
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

STRATS=("random" "genomic" "prm_1pct" "prm_20pct" "motif_grammar")
SIZES=(1000000 2000000)

STRAT_IDX=$((T / 2))
SIZE_IDX=$((T % 2))
STRAT=${STRATS[$STRAT_IDX]}
SIZE=${SIZES[$SIZE_IDX]}
SEED=${OPTUNA_SEED:-42}

echo "=== Optuna 1M+: ${STRAT} n=${SIZE} seed=${SEED} — $(date) ==="

uv run --no-sync python scripts/optuna_legnet_scaling.py \
    --strategy "${STRAT}" \
    --size "${SIZE}" \
    --seed "${SEED}" \
    --n-trials 10

echo "=== DONE — $(date) ==="
