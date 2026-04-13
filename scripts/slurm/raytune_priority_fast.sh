#!/bin/bash
# PRIORITY RayTune HP sweep on FAST queue (4h, highest priority).
# Peter's talk is Tuesday — these need to finish ASAP.
#
# 6 strategies × 5 small sizes = 30 jobs
# Split across fast (2 GPU) and default (4 GPU) queues
#
# This script: fast queue (first 12 jobs)
#
#SBATCH --job-name=rt_fast
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=fast
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

set -euo pipefail
set +u; source /etc/profile.d/modules.sh; set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD"
source scripts/slurm/setup_hpc_deps.sh

T=$SLURM_ARRAY_TASK_ID

# 6 strategies, 5 sizes — but split: fast gets the smallest (fastest) sizes
STRATS=("random" "genomic" "prm_1pct" "prm_20pct" "motif_grammar" "evoaug_structural")
SIZES=(1000 5000 10000 20000 50000)

STRAT_IDX=$((T / 5))
SIZE_IDX=$((T % 5))
STRAT=${STRATS[$STRAT_IDX]}
SIZE=${SIZES[$SIZE_IDX]}

# Skip if result already exists
OUT_DIR="outputs/raytune_best/${STRAT}/n${SIZE}"
[ -f "${OUT_DIR}/best_config_seed42.json" ] && echo "SKIP" && exit 0

echo "=== RayTune FAST: ${STRAT} n=${SIZE} — $(date) ==="

uv run --no-sync python scripts/raytune_legnet_scaling.py \
    --strategy "${STRAT}" \
    --size "${SIZE}" \
    --seed 42 \
    --n-trials 20 \
    --cpus 8

echo "=== DONE — $(date) ==="
