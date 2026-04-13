#!/bin/bash
# Optuna HP search for LARGE sizes: 100K, 200K, 500K.
#
# 6 strategies × 3 sizes = 18 jobs per seed.
# Array: strat_idx * 3 + size_idx (0-17)
#
#SBATCH --job-name=opt_lrg
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

STRATS=("random" "genomic" "prm_1pct" "prm_20pct" "motif_grammar" "evoaug_heavy")
SIZES=(100000 200000 500000)

STRAT_IDX=$((T / 3))
SIZE_IDX=$((T % 3))
STRAT=${STRATS[$STRAT_IDX]}
SIZE=${SIZES[$SIZE_IDX]}
SEED=${OPTUNA_SEED:-42}

echo "=== Optuna large: ${STRAT} n=${SIZE} seed=${SEED} — $(date) ==="

uv run --no-sync python scripts/optuna_legnet_scaling.py \
    --strategy "${STRAT}" \
    --size "${SIZE}" \
    --seed "${SEED}" \
    --n-trials 10

echo "=== DONE — $(date) ==="
