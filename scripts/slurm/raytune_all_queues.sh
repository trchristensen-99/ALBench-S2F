#!/bin/bash
# RayTune on ALL available queues simultaneously.
# Each job runs 10 Optuna trials for one (strategy, size).
#
# Array mapping (same as peter6):
#   T = strat_idx * 5 + size_idx
#   strats: random(0), genomic(1), prm_1pct(2), prm_20pct(3), motif_grammar(4), evoaug_heavy(5)
#   sizes: 1000(0), 5000(1), 10000(2), 20000(3), 50000(4)
#
#SBATCH --job-name=rt_all
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
SIZES=(1000 5000 10000 20000 50000)

STRAT_IDX=$((T / 5))
SIZE_IDX=$((T % 5))
STRAT=${STRATS[$STRAT_IDX]}
SIZE=${SIZES[$SIZE_IDX]}

echo "=== RayTune: ${STRAT} n=${SIZE} — $(date) ==="

uv run --no-sync python scripts/raytune_legnet_scaling.py \
    --strategy "${STRAT}" \
    --size "${SIZE}" \
    --seed 42 \
    --n-trials 10 \
    --cpus 8

echo "=== DONE — $(date) ==="
