#!/bin/bash
# Joint capacity × regularization × LR schedule sweep.
# Tests PI's recommendations: depth/width interact with dataset size,
# dropout + weight_decay for regularization, pct_start for schedule.
#
# Design: test each factor independently at 2 sizes, 2 strategies.
# Then test key interactions (capacity × reg, capacity × size).
#
# Phase 1 (0-23):  Depth sweep (4,6,8,10 blocks) × 2 sizes × 3 strats
# Phase 2 (24-47): Width sweep (narrow,default,wide) × 2 sizes × 3 strats + ks=7
# Phase 3 (48-71): Dropout sweep (0, 0.05, 0.1, 0.2) × 2 sizes × 3 strats
# Phase 4 (72-89): Weight decay sweep (0.001, 0.01, 0.03, 0.1) at default arch × n=5K,100K
# Phase 5 (90-107): pct_start sweep (0.1, 0.2, 0.3, 0.5) × 2 sizes × 3 strats
# Phase 6 (108-131): Key interactions: wide+dropout at large N, narrow+low_wd at small N
#
# Total: 132 jobs (many are fast — small N or small arch)
#
#SBATCH --job-name=hp_cap
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

S3=("random" "evoaug_heavy" "genomic")
SZ2=(5000 100000)

LR=0.003; BS=256; DROPOUT=0.0; WD=""; PCT=""; ARCH_ARGS=""
HP_TAG=""

# ── Phase 1: Depth sweep (0-23) ──
if [ $T -lt 24 ]; then
    IDX=$T
    HP_IDX=$((IDX / 6)); SIZE_IDX=$(( (IDX % 6) / 3 )); STRAT_IDX=$((IDX % 3))
    STRAT=${S3[$STRAT_IDX]}; SIZE=${SZ2[$SIZE_IDX]}
    case $HP_IDX in
        0) ARCH_ARGS="--block-sizes 256,128,64,32"; HP_TAG="depth4" ;;
        1) ARCH_ARGS="--block-sizes 256,256,128,64,64,32"; HP_TAG="depth6" ;;
        2) HP_TAG="depth8_default" ;;  # default 8 blocks, no override
        3) ARCH_ARGS="--block-sizes 256,256,256,128,128,64,64,64,32,32"; HP_TAG="depth10" ;;
    esac

# ── Phase 2: Width + ks sweep (24-47) ──
elif [ $T -lt 48 ]; then
    IDX=$((T - 24))
    HP_IDX=$((IDX / 6)); SIZE_IDX=$(( (IDX % 6) / 3 )); STRAT_IDX=$((IDX % 3))
    STRAT=${S3[$STRAT_IDX]}; SIZE=${SZ2[$SIZE_IDX]}
    case $HP_IDX in
        0) ARCH_ARGS="--block-sizes 128,128,64,64,32,32,16,16"; HP_TAG="narrow" ;;
        1) HP_TAG="default_width" ;;  # default
        2) ARCH_ARGS="--block-sizes 512,512,256,256,128,128,64,64"; HP_TAG="wide" ;;
        3) ARCH_ARGS="--ks 7"; HP_TAG="ks7" ;;
    esac

# ── Phase 3: Dropout sweep (48-71) ──
elif [ $T -lt 72 ]; then
    IDX=$((T - 48))
    HP_IDX=$((IDX / 6)); SIZE_IDX=$(( (IDX % 6) / 3 )); STRAT_IDX=$((IDX % 3))
    STRAT=${S3[$STRAT_IDX]}; SIZE=${SZ2[$SIZE_IDX]}
    DROPOUT_VALS=(0.0 0.05 0.1 0.2)
    DROPOUT=${DROPOUT_VALS[$HP_IDX]}
    HP_TAG="dropout${DROPOUT}"

# ── Phase 4: Weight decay sweep (72-95) ──
elif [ $T -lt 96 ]; then
    IDX=$((T - 72))
    HP_IDX=$((IDX / 6)); SIZE_IDX=$(( (IDX % 6) / 3 )); STRAT_IDX=$((IDX % 3))
    STRAT=${S3[$STRAT_IDX]}; SIZE=${SZ2[$SIZE_IDX]}
    WD_VALS=(0.001 0.01 0.03 0.1)
    WD="--weight-decay ${WD_VALS[$HP_IDX]}"
    HP_TAG="wd${WD_VALS[$HP_IDX]}"

# ── Phase 5: pct_start sweep (96-119) ──
elif [ $T -lt 120 ]; then
    IDX=$((T - 96))
    HP_IDX=$((IDX / 6)); SIZE_IDX=$(( (IDX % 6) / 3 )); STRAT_IDX=$((IDX % 3))
    STRAT=${S3[$STRAT_IDX]}; SIZE=${SZ2[$SIZE_IDX]}
    PCT_VALS=(0.1 0.2 0.3 0.5)
    PCT="--pct-start ${PCT_VALS[$HP_IDX]}"
    HP_TAG="pct${PCT_VALS[$HP_IDX]}"

# ── Phase 6: Key interactions (120-143) ──
elif [ $T -lt 144 ]; then
    IDX=$((T - 120))
    HP_IDX=$((IDX / 6)); SIZE_IDX=$(( (IDX % 6) / 3 )); STRAT_IDX=$((IDX % 3))
    STRAT=${S3[$STRAT_IDX]}; SIZE=${SZ2[$SIZE_IDX]}
    case $HP_IDX in
        0) ARCH_ARGS="--block-sizes 512,512,256,256,128,128,64,64"; DROPOUT=0.1; HP_TAG="wide_drop01" ;;
        1) ARCH_ARGS="--block-sizes 512,512,256,256,128,128,64,64"; DROPOUT=0.2; HP_TAG="wide_drop02" ;;
        2) ARCH_ARGS="--block-sizes 128,128,64,64,32,32,16,16"; WD="--weight-decay 0.001"; HP_TAG="narrow_wd001" ;;
        3) ARCH_ARGS="--block-sizes 256,256,256,128,128,64,64,64,32,32"; DROPOUT=0.1; HP_TAG="depth10_drop01" ;;
    esac

else
    echo "SKIP: T=$T out of range"; exit 0
fi

POOL_DIR="outputs/labeled_pools_5m/k562/ag_s2"
[ ! -f "${POOL_DIR}/${STRAT}/pool.npz" ] && POOL_DIR="outputs/labeled_pools_2m/k562/ag_s2"
[ ! -f "${POOL_DIR}/${STRAT}/pool.npz" ] && POOL_DIR="outputs/labeled_pools/k562/ag_s2"

OUT="outputs/hp_capacity_sweep/k562/legnet_ag_s2"
SEED=$(python3 -c "import random; random.seed($T * 100003 + 555); print(random.randint(1, 999999999))")
RESULT_DIR="${OUT}/${STRAT}/n${SIZE}/${HP_TAG}"

EXISTING=$(find "${RESULT_DIR}" -name "result.json" 2>/dev/null | head -1 || true)
if [ -n "${EXISTING}" ]; then echo "SKIP"; exit 0; fi

sleep $((T % 7))
echo "=== Capacity sweep: ${STRAT} n=${SIZE} ${HP_TAG} — $(date) ==="

# Build dropout arg
DROPOUT_ARG=""
if [ "$(echo "$DROPOUT > 0" | bc -l 2>/dev/null || python3 -c "print(1 if $DROPOUT > 0 else 0)")" = "1" ]; then
    DROPOUT_ARG="--dropout ${DROPOUT}"
fi

uv run --no-sync python experiments/exp1_1_scaling.py \
    --task k562 --student legnet --oracle ag_s2 \
    --reservoir "${STRAT}" \
    --pool-base-dir "${POOL_DIR}" \
    --n-replicates 1 --seed "${SEED}" \
    --output-dir "${RESULT_DIR}" \
    --training-sizes "${SIZE}" \
    --chr-split --lr "${LR}" --batch-size "${BS}" \
    --epochs 80 --ensemble-size 1 --early-stop-patience 10 \
    ${ARCH_ARGS} ${DROPOUT_ARG} ${WD} ${PCT}

echo "=== DONE — $(date) ==="
