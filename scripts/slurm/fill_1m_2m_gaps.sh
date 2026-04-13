#!/bin/bash
# Fill 1M and 2M gaps in scaling curves.
#
# All strategies are missing 1M and 2M. Uses 2M pools (sampling with
# replacement for 2M = full pool, for 1M = subsample).
#
# Strategies with 2M pools: random, dinuc_shuffle, gc_matched, prm_5pct,
# prm_10pct, evoaug_structural, evoaug_heavy, recombination_uniform,
# motif_planted, motif_grammar
#
# 10 strategies × 2 sizes × 3 seeds = 60 jobs, but use best HP only
# (bs=1024, lr=0.005) to keep it manageable = 10 × 2 × 1 = 20 jobs first
#
# Array: strategy_idx * 2 + size_idx
#   0-1:   random        1M, 2M
#   2-3:   dinuc_shuffle  1M, 2M
#   4-5:   evoaug_structural 1M, 2M
#   6-7:   prm_5pct      1M, 2M
#   8-9:   motif_grammar  1M, 2M
#  10-11:  motif_planted  1M, 2M
#  12-13:  recombination_uniform 1M, 2M
#  14-15:  evoaug_heavy   1M, 2M
#  16-17:  gc_matched     1M, 2M
#  18-19:  prm_10pct      1M, 2M
#
# Submit:
#   /cm/shared/apps/slurm/current/bin/sbatch --array=0-19 scripts/slurm/fill_1m_2m_gaps.sh
#
#SBATCH --job-name=1m2m_gap
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --time=48:00:00
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

STRATS=("random" "random" "dinuc_shuffle" "dinuc_shuffle" "evoaug_structural" "evoaug_structural" "prm_5pct" "prm_5pct" "motif_grammar" "motif_grammar" "motif_planted" "motif_planted" "recombination_uniform" "recombination_uniform" "evoaug_heavy" "evoaug_heavy" "gc_matched" "gc_matched" "prm_10pct" "prm_10pct")
SIZES=(1000000 2000000 1000000 2000000 1000000 2000000 1000000 2000000 1000000 2000000 1000000 2000000 1000000 2000000 1000000 2000000 1000000 2000000 1000000 2000000)

STRAT=${STRATS[$T]}
SIZE=${SIZES[$T]}
SEED=42

OUT="outputs/exp1_1/k562/legnet_ag_s2"

[ -f "${OUT}/${STRAT}/n${SIZE}/hp0/seed${SEED}/result.json" ] && echo "SKIP" && exit 0

POOL_DIR="outputs/labeled_pools_2m/k562/ag_s2"

echo "=== ${STRAT} n=${SIZE} seed=${SEED} — $(date) ==="

uv run --no-sync python experiments/exp1_1_scaling.py \
    --task k562 --student legnet --oracle ag_s2 \
    --reservoir "${STRAT}" \
    --pool-base-dir "${POOL_DIR}" \
    --n-replicates 1 --seed "${SEED}" \
    --output-dir "${OUT}" \
    --training-sizes "${SIZE}" \
    --chr-split --lr 0.005 --batch-size 1024 \
    --epochs 80 --ensemble-size 1 --early-stop-patience 10

echo "=== DONE — $(date) ==="
