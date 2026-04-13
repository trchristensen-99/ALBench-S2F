#!/bin/bash
# PRIORITY: 6-strategy scaling curves for Peter's talk.
#
# 6 strategies × 8 sizes × 3 seeds × 2 HP configs = 288 jobs
# But we'll do: 6 strats × 8 sizes × 1 seed × 2 HPs = 96 jobs first
# Then 2 more seeds on fast queue as follow-up
#
# HP configs: lr=0.001/bs=1024 (better at large N) + lr=0.005/bs=1024 (legacy)
#
# Strategies: random, genomic, prm_1pct, prm_20pct, motif_grammar, evoaug_structural
# Sizes: 1000, 5000, 10000, 20000, 50000, 100000, 200000, 500000
#
# Array: strat_idx * 16 + size_idx * 2 + hp_idx
# = 6 * 16 = 96 jobs
#
#SBATCH --job-name=p6_scal
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

STRATS=("random" "genomic" "prm_1pct" "prm_20pct" "motif_grammar" "evoaug_structural")
SIZES=(1000 5000 10000 20000 50000 100000 200000 500000)
LRS=("0.001" "0.005")
BSS=("1024" "1024")

STRAT_IDX=$((T / 16))
SIZE_IDX=$(( (T % 16) / 2 ))
HP_IDX=$((T % 2))

STRAT=${STRATS[$STRAT_IDX]}
SIZE=${SIZES[$SIZE_IDX]}
LR=${LRS[$HP_IDX]}
BS=${BSS[$HP_IDX]}
SEED=42

OUT="outputs/exp1_1_peter6/k562/legnet_ag_s2"

[ -f "${OUT}/${STRAT}/n${SIZE}/hp${HP_IDX}/seed${SEED}/result.json" ] && echo "SKIP" && exit 0

POOL_DIR="outputs/labeled_pools_2m/k562/ag_s2"
[ ! -f "${POOL_DIR}/${STRAT}/pool.npz" ] && POOL_DIR="outputs/labeled_pools/k562/ag_s2"

echo "=== ${STRAT} n=${SIZE} lr=${LR} bs=${BS} seed=${SEED} — $(date) ==="

uv run --no-sync python experiments/exp1_1_scaling.py \
    --task k562 --student legnet --oracle ag_s2 \
    --reservoir "${STRAT}" \
    --pool-base-dir "${POOL_DIR}" \
    --n-replicates 1 --seed "${SEED}" \
    --output-dir "${OUT}" \
    --training-sizes "${SIZE}" \
    --chr-split --lr "${LR}" --batch-size "${BS}" \
    --epochs 80 --ensemble-size 1 --early-stop-patience 10

echo "=== DONE — $(date) ==="
