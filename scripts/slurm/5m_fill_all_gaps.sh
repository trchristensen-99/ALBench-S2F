#!/bin/bash
# Fill ALL remaining 5M scaling gaps.
#
# Uses labeled_pools_2m where available, falls back to labeled_pools.
# 5M with 2M pool = sampling with replacement (~2.5x oversampling).
#
# Gaps to fill:
#   0: evoaug_structural seed=42    (from 2m pool)
#   1: evoaug_structural seed=1042
#   2: evoaug_structural seed=2042
#   3: dinuc_shuffle seed=42        (from 2m pool)
#   4: dinuc_shuffle seed=1042
#   5: dinuc_shuffle seed=2042
#   6: prm_5pct seed=1042           (from 2m pool, seed 42 done)
#   7: prm_5pct seed=2042
#   8: random seed=2042             (resubmitted separately but add backup)
#   9: genomic seed=42              (618K pool — 8x oversample)
#  10: genomic seed=1042
#  11: genomic seed=2042
#
# Submit:
#   /cm/shared/apps/slurm/current/bin/sbatch --array=0-11 scripts/slurm/5m_fill_all_gaps.sh
#
#SBATCH --job-name=5m_gaps
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

STRATS=("evoaug_structural" "evoaug_structural" "evoaug_structural" "dinuc_shuffle" "dinuc_shuffle" "dinuc_shuffle" "prm_5pct" "prm_5pct" "random" "genomic" "genomic" "genomic")
SEEDS=(42 1042 2042 42 1042 2042 1042 2042 2042 42 1042 2042)

STRAT=${STRATS[$T]}
SEED=${SEEDS[$T]}

OUT="outputs/exp1_1_5m_scaling/k562/legnet_ag_s2"

[ -f "${OUT}/${STRAT}/n5000000/hp0/seed${SEED}/result.json" ] && echo "SKIP: ${STRAT} seed=${SEED}" && exit 0

# Choose pool: 2m if available, otherwise original
if [ -f "outputs/labeled_pools_2m/k562/ag_s2/${STRAT}/pool.npz" ]; then
    POOL_DIR="outputs/labeled_pools_2m/k562/ag_s2"
else
    POOL_DIR="outputs/labeled_pools/k562/ag_s2"
fi

echo "=== 5M ${STRAT} seed=${SEED} pool=${POOL_DIR} — $(date) ==="

uv run --no-sync python experiments/exp1_1_scaling.py \
    --task k562 --student legnet --oracle ag_s2 \
    --reservoir "${STRAT}" \
    --pool-base-dir "${POOL_DIR}" \
    --n-replicates 1 --seed "${SEED}" \
    --output-dir "${OUT}" \
    --training-sizes 5000000 \
    --chr-split --lr 0.005 --batch-size 1024 \
    --epochs 80 --ensemble-size 1 --early-stop-patience 10

echo "=== DONE — $(date) ==="
