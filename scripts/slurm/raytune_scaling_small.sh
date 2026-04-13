#!/bin/bash
# RayTune HP sweep for 6 priority strategies at small sizes (1K-50K).
#
# Peter's 6 priority strategies:
#   random, genomic, prm_1pct, prm_20pct, motif_grammar, evoaug_structural
#
# Sizes: 1000, 5000, 10000, 20000, 50000
# 6 strategies × 5 sizes = 30 jobs
# Each runs 20 RayTune trials with ASHA early stopping
#
# Submit:
#   /cm/shared/apps/slurm/current/bin/sbatch --array=0-29 scripts/slurm/raytune_scaling_small.sh
#
#SBATCH --job-name=rt_small
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --time=06:00:00
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

STRATS=("random" "genomic" "prm_1pct" "prm_20pct" "motif_grammar" "evoaug_structural")
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
    --n-trials 20 \
    --cpus 8

echo "=== DONE — $(date) ==="
