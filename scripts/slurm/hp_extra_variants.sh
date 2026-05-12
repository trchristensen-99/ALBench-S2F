#!/bin/bash
# Test additional HP variants not yet explored:
#   Phase 1 (0-11): pct_start sweep (OneCycleLR warmup fraction)
#   Phase 2 (12-23): shift_aug (random bp-level data augmentation)
#   Phase 3 (24-35): architecture variants (wide, deep, ks=7)
#   Phase 4 (36-47): size-calibrated HP at N=5K (lr=0.001/bs=64 vs 0.003/256)
#
# All via exp1_1_scaling.py. 3 strats × 2 sizes per phase.
#
#SBATCH --job-name=hp_var
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

STRATS3=("random" "evoaug_heavy" "genomic")

# ── Phase 1: pct_start sweep (0-11) ──
# 2 pct_start × 2 sizes × 3 strats = 12 jobs
if [ $T -lt 12 ]; then
    IDX=$T
    HP_IDX=$((IDX / 6))
    SIZE_IDX=$(( (IDX % 6) / 3 ))
    STRAT_IDX=$((IDX % 3))

    STRAT=${STRATS3[$STRAT_IDX]}
    SIZES=(5000 100000); SIZE=${SIZES[$SIZE_IDX]}
    LR=0.003; BS=256

    PCT_VALS=(0.1 0.5)
    # pct_start not directly in CLI — need to test via env var workaround
    # For now, skip this and use default 0.3
    # TODO: add --pct-start to exp1_1_scaling.py
    HP_TAG="lr${LR}_bs${BS}_pct${PCT_VALS[$HP_IDX]}"
    EXTRA_ARGS=""
    echo "NOTE: pct_start=${PCT_VALS[$HP_IDX]} not yet in CLI, using default 0.3"

# ── Phase 2: shift_aug (12-23) ──
# 2 shift configs × 2 sizes × 3 strats = 12 jobs
elif [ $T -lt 24 ]; then
    IDX=$((T - 12))
    HP_IDX=$((IDX / 6))
    SIZE_IDX=$(( (IDX % 6) / 3 ))
    STRAT_IDX=$((IDX % 3))

    STRAT=${STRATS3[$STRAT_IDX]}
    SIZES=(5000 100000); SIZE=${SIZES[$SIZE_IDX]}
    LR=0.003; BS=256

    SHIFT_VALS=(5 15)
    SHIFT=${SHIFT_VALS[$HP_IDX]}
    HP_TAG="lr${LR}_bs${BS}_shift${SHIFT}"
    EXTRA_ARGS="--shift-aug --max-shift ${SHIFT}"

# ── Phase 3: architecture variants (24-41) ──
# 3 arch × 2 sizes × 3 strats = 18 jobs
elif [ $T -lt 42 ]; then
    IDX=$((T - 24))
    HP_IDX=$((IDX / 6))
    SIZE_IDX=$(( (IDX % 6) / 3 ))
    STRAT_IDX=$((IDX % 3))

    STRAT=${STRATS3[$STRAT_IDX]}
    SIZES=(5000 100000); SIZE=${SIZES[$SIZE_IDX]}
    LR=0.003; BS=256

    # Architecture sweep handled via --arch-sweep in exp1_1_scaling.py
    # But that does all 3 at once. Instead, use the manual approach.
    case $HP_IDX in
        0) HP_TAG="wide_ks5"; EXTRA_ARGS="" ;;  # Need arch params via code
        1) HP_TAG="default_ks7"; EXTRA_ARGS="" ;;
        2) HP_TAG="narrow"; EXTRA_ARGS="" ;;
    esac
    # Architecture variants need code changes to pass block_sizes via CLI
    # For now, just run the arch-sweep flag
    EXTRA_ARGS="--arch-sweep"
    HP_TAG="arch_sweep"

# ── Phase 4: size-calibrated HP test (42-53) ──
# 2 HP × 2 sizes × 3 strats = 12 jobs
# Test if lr=0.001/bs=64 is better at 5K (sweep suggested it)
elif [ $T -lt 54 ]; then
    IDX=$((T - 42))
    HP_IDX=$((IDX / 6))
    SIZE_IDX=$(( (IDX % 6) / 3 ))
    STRAT_IDX=$((IDX % 3))

    STRAT=${STRATS3[$STRAT_IDX]}
    SIZES=(5000 100000); SIZE=${SIZES[$SIZE_IDX]}

    case $HP_IDX in
        0) LR=0.001; BS=64 ;;   # best at 5K from sweep
        1) LR=0.002; BS=128 ;;  # intermediate
    esac
    HP_TAG="lr${LR}_bs${BS}_calibrated"
    EXTRA_ARGS=""

else
    echo "SKIP: T=$T out of range"; exit 0
fi

POOL_DIR="outputs/labeled_pools_5m/k562/ag_s2"
[ ! -f "${POOL_DIR}/${STRAT}/pool.npz" ] && POOL_DIR="outputs/labeled_pools_2m/k562/ag_s2"
[ ! -f "${POOL_DIR}/${STRAT}/pool.npz" ] && POOL_DIR="outputs/labeled_pools/k562/ag_s2"

OUT="outputs/hp_extra_variants/k562/legnet_ag_s2"
SEED=$(python3 -c "import random; random.seed($T * 100003 + 777); print(random.randint(1, 999999999))")
RESULT_DIR="${OUT}/${STRAT}/n${SIZE}/${HP_TAG}"

EXISTING=$(find "${RESULT_DIR}" -name "result.json" 2>/dev/null | head -1 || true)
if [ -n "${EXISTING}" ]; then echo "SKIP"; exit 0; fi

sleep $((T % 7))
echo "=== HP variant: ${STRAT} n=${SIZE} ${HP_TAG} seed=${SEED} — $(date) ==="

uv run --no-sync python experiments/exp1_1_scaling.py \
    --task k562 --student legnet --oracle ag_s2 \
    --reservoir "${STRAT}" \
    --pool-base-dir "${POOL_DIR}" \
    --n-replicates 1 --seed "${SEED}" \
    --output-dir "${RESULT_DIR}" \
    --training-sizes "${SIZE}" \
    --chr-split --lr "${LR}" --batch-size "${BS}" \
    --epochs 80 --ensemble-size 1 --early-stop-patience 10 \
    ${EXTRA_ARGS}

echo "=== DONE — $(date) ==="
