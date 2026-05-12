#!/bin/bash
# Overnight HP exploration via exp1_1_scaling.py (identical pipeline to definitive).
#
# Phase 1 (0-23): Fine LR grid at bs=256 + promising LR/BS combos
# Phase 2 (24-41): Best HP with 3 reps across all 5 strategies at 5K+100K
# Phase 3 (42-59): Higher LR with large BS (linear scaling rule)
# Phase 4 (60-71): Extra sizes (20K, 50K) at best HP, 3 strategies
#
# Array: varies by phase (see indexing below)
# Total: 72 configs, many will produce multiple runs
#
#SBATCH --job-name=hp_ovn
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
export TORCHDYNAMO_DISABLE=1
source scripts/slurm/setup_hpc_deps.sh

T=$SLURM_ARRAY_TASK_ID

# ── Phase 1: Fine LR grid + promising combos (0-23) ──
# 8 HP configs × 3 strats (random, evoaug_heavy, genomic)
if [ $T -lt 24 ]; then
    HP_IDX=$((T / 3))
    STRAT_IDX=$((T % 3))
    STRATS=("random" "evoaug_heavy" "genomic")
    STRAT=${STRATS[$STRAT_IDX]}
    SIZE=100000  # focus on 100K where differences are clearest

    case $HP_IDX in
        0) LR=0.002;  BS=256 ;;  # slightly below current best
        1) LR=0.004;  BS=256 ;;  # slightly above
        2) LR=0.006;  BS=256 ;;  # further above
        3) LR=0.001;  BS=256 ;;  # lower end
        4) LR=0.008;  BS=256 ;;  # high
        5) LR=0.002;  BS=128 ;;  # promising from old Optuna
        6) LR=0.004;  BS=128 ;;  # bracket at 128
        7) LR=0.005;  BS=128 ;;  # Optuna found ~0.004/128 good at 200K
    esac
    HP_TAG="lr${LR}_bs${BS}"
    PAT=10

# ── Phase 2: Best HP (0.003/256) with 3 reps × all 5 strategies (24-53) ──
# 5 strats × 2 sizes × 3 reps = 30 jobs
elif [ $T -lt 54 ]; then
    IDX=$((T - 24))
    STRAT_IDX=$((IDX / 6))
    SIZE_IDX=$(( (IDX % 6) / 3 ))
    REP=$((IDX % 3))
    STRATS=("random" "prm_1pct" "prm_20pct" "motif_grammar" "evoaug_heavy")
    STRAT=${STRATS[$STRAT_IDX]}
    SIZES=(5000 100000)
    SIZE=${SIZES[$SIZE_IDX]}
    LR=0.003; BS=256; PAT=10
    HP_TAG="lr${LR}_bs${BS}_rep${REP}"

# ── Phase 3: High LR with large BS — linear scaling rule (54-71) ──
# 6 HP configs × 3 strats at 100K
elif [ $T -lt 72 ]; then
    IDX=$((T - 54))
    HP_IDX=$((IDX / 3))
    STRAT_IDX=$((IDX % 3))
    STRATS=("random" "evoaug_heavy" "genomic")
    STRAT=${STRATS[$STRAT_IDX]}
    SIZE=100000

    case $HP_IDX in
        0) LR=0.006;  BS=512  ;;  # 2x bs → 2x lr
        1) LR=0.010;  BS=512  ;;  # aggressive
        2) LR=0.012;  BS=1024 ;;  # 4x bs → 4x lr
        3) LR=0.015;  BS=1024 ;;  # more aggressive
        4) LR=0.020;  BS=2048 ;;  # 8x bs → 8x lr (2048 ok at 100K: 100K/32=3125)
        5) LR=0.004;  BS=512  ;;  # moderate scaling
    esac
    HP_TAG="lr${LR}_bs${BS}"
    PAT=10

# ── Phase 4: Extra sizes at best HP (72-89) ──
# 3 strats × 2 sizes × 3 reps = 18 jobs
elif [ $T -lt 90 ]; then
    IDX=$((T - 72))
    STRAT_IDX=$((IDX / 6))
    SIZE_IDX=$(( (IDX % 6) / 3 ))
    REP=$((IDX % 3))
    STRATS=("random" "evoaug_heavy" "genomic")
    STRAT=${STRATS[$STRAT_IDX]}
    SIZES=(20000 50000)
    SIZE=${SIZES[$SIZE_IDX]}
    LR=0.003; BS=256; PAT=10
    HP_TAG="lr${LR}_bs${BS}_n${SIZE}_rep${REP}"

else
    echo "SKIP: T=$T out of range"
    exit 0
fi

# BS cap
MAX_BS=$((SIZE / 32))
[ $BS -gt $MAX_BS ] && echo "SKIP: bs=$BS > n/32=$MAX_BS" && exit 0

POOL_DIR="outputs/labeled_pools_5m/k562/ag_s2"
[ ! -f "${POOL_DIR}/${STRAT}/pool.npz" ] && POOL_DIR="outputs/labeled_pools_2m/k562/ag_s2"
[ ! -f "${POOL_DIR}/${STRAT}/pool.npz" ] && POOL_DIR="outputs/labeled_pools/k562/ag_s2"

OUT="outputs/hp_overnight/k562/legnet_ag_s2"

SEED=$(python3 -c "import random; random.seed($T * 100003 + 99); print(random.randint(1, 999999999))")
RESULT_DIR="${OUT}/${STRAT}/n${SIZE}/hp_${HP_TAG}"

# Skip check: look for any result.json under this dir
EXISTING=$(find "${RESULT_DIR}" -name "result.json" 2>/dev/null | head -1 || true)
if [ -n "${EXISTING}" ]; then echo "SKIP"; exit 0; fi

sleep $((T % 7))
echo "=== HP overnight: ${STRAT} n=${SIZE} ${HP_TAG} seed=${SEED} — $(date) ==="

uv run --no-sync python experiments/exp1_1_scaling.py \
    --task k562 --student legnet --oracle ag_s2 \
    --reservoir "${STRAT}" \
    --pool-base-dir "${POOL_DIR}" \
    --n-replicates 1 --seed "${SEED}" \
    --output-dir "${RESULT_DIR}" \
    --training-sizes "${SIZE}" \
    --chr-split --lr "${LR}" --batch-size "${BS}" \
    --epochs 80 --ensemble-size 1 --early-stop-patience "${PAT}"

echo "=== DONE — $(date) ==="
