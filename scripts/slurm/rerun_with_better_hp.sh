#!/bin/bash
# Rerun 100K-500K with the better HP config (lr=0.001, bs=1024) that was
# found to work well at 1M+. This creates a consistent scaling curve
# without the HP discontinuity at the 500K→1M boundary.
#
# 10 strategies × 3 sizes (100K, 200K, 500K) × 3 seeds = 90 jobs
# But start with seed=42 only for speed = 30 jobs
#
# The existing data at 100K-500K used lr=0.005,bs=1024 (from small-N HP sweep)
# The 1M/2M data used lr=0.001,bs=512 or bs=2048
# We'll use lr=0.001,bs=1024 for consistency
#
#SBATCH --job-name=hp_rerun
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --time=12:00:00
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

# 10 strategies × 3 sizes, all seed=42
STRATS=("random" "genomic" "dinuc_shuffle" "evoaug_structural" "evoaug_heavy" "prm_5pct" "motif_grammar" "motif_planted" "recombination_uniform" "gc_matched")
SIZES=(100000 200000 500000)

STRAT_IDX=$((T / 3))
SIZE_IDX=$((T % 3))
STRAT=${STRATS[$STRAT_IDX]}
SIZE=${SIZES[$SIZE_IDX]}
SEED=42

OUT="outputs/exp1_1_hp_rerun/k562/legnet_ag_s2"

[ -f "${OUT}/${STRAT}/n${SIZE}/hp0/seed${SEED}/result.json" ] && echo "SKIP" && exit 0

POOL_DIR="outputs/labeled_pools_2m/k562/ag_s2"
[ ! -f "${POOL_DIR}/${STRAT}/pool.npz" ] && POOL_DIR="outputs/labeled_pools/k562/ag_s2"

echo "=== ${STRAT} n=${SIZE} seed=${SEED} lr=0.001 — $(date) ==="

uv run --no-sync python experiments/exp1_1_scaling.py \
    --task k562 --student legnet --oracle ag_s2 \
    --reservoir "${STRAT}" \
    --pool-base-dir "${POOL_DIR}" \
    --n-replicates 1 --seed "${SEED}" \
    --output-dir "${OUT}" \
    --training-sizes "${SIZE}" \
    --chr-split --lr 0.001 --batch-size 1024 \
    --epochs 80 --ensemble-size 1 --early-stop-patience 10

echo "=== DONE — $(date) ==="
