#!/bin/bash
# HP sweep using the SAME exp1_1_scaling.py code as the definitive.
# This ensures identical training + evaluation pipeline.
# Only the HP args (lr, bs) and output dir differ.
#
# 14 HP configs × 2 sizes × 3 strategies = 84 jobs
#   0-11: LR/BS grid (same 12 configs as hp_grid_poolval)
#   12-13: patience sweep (5, 20 at lr=0.003/bs=256)
#
# Array: HP_IDX * 6 + SIZE_IDX * 3 + STRAT_IDX
#
#SBATCH --job-name=hp_exp
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

STRATS=("random" "evoaug_heavy" "genomic")
SIZES=(5000 100000)

HP_IDX=$((T / 6))
SIZE_IDX=$(( (T % 6) / 3 ))
STRAT_IDX=$((T % 3))

STRAT=${STRATS[$STRAT_IDX]}
SIZE=${SIZES[$SIZE_IDX]}

# HP configs
if [ $HP_IDX -lt 12 ]; then
    # LR/BS grid
    LRS=(0.0005 0.001  0.001 0.002  0.001 0.003  0.002 0.003 0.005  0.003 0.005 0.008)
    BSS=(32     32     64    64     128   128    256   256   256    512   512   512)
    LR=${LRS[$HP_IDX]}
    BS=${BSS[$HP_IDX]}
    EXTRA_ARGS=""
    HP_TAG="lr${LR}_bs${BS}"
elif [ $HP_IDX -lt 14 ]; then
    # Patience sweep at lr=0.003/bs=256
    PAT_IDX=$((HP_IDX - 12))
    PAT_VALS=(5 20)
    LR=0.003; BS=256
    PAT_OVERRIDE=${PAT_VALS[$PAT_IDX]}
    EXTRA_ARGS=""
    HP_TAG="lr${LR}_bs${BS}_pat${PAT_OVERRIDE}"
else
    echo "SKIP: invalid HP_IDX=$HP_IDX"
    exit 0
fi

# BS cap: skip if bs > n/32
MAX_BS=$((SIZE / 32))
[ $BS -gt $MAX_BS ] && echo "SKIP: bs=$BS > n/32=$MAX_BS" && exit 0

POOL_DIR="outputs/labeled_pools_5m/k562/ag_s2"
[ ! -f "${POOL_DIR}/${STRAT}/pool.npz" ] && POOL_DIR="outputs/labeled_pools_2m/k562/ag_s2"
[ ! -f "${POOL_DIR}/${STRAT}/pool.npz" ] && POOL_DIR="outputs/labeled_pools/k562/ag_s2"

OUT="outputs/hp_sweep_exp/k562/legnet_ag_s2"

# Check if result already exists (exp1_1_scaling saves to hp0/seed*/result.json)
SEED=$(python3 -c "import random; random.seed($T * 100003 + 42); print(random.randint(1, 999999999))")
RESULT="${OUT}/${STRAT}/n${SIZE}/hp${HP_IDX}_${HP_TAG}/hp0/seed${SEED}/result.json"
[ -f "${RESULT}" ] && echo "SKIP" && exit 0

sleep $((T % 7))
echo "=== HP exp: ${STRAT} n=${SIZE} ${HP_TAG} seed=${SEED} — $(date) ==="

PAT=${PAT_OVERRIDE:-10}

uv run --no-sync python experiments/exp1_1_scaling.py \
    --task k562 --student legnet --oracle ag_s2 \
    --reservoir "${STRAT}" \
    --pool-base-dir "${POOL_DIR}" \
    --n-replicates 1 --seed "${SEED}" \
    --output-dir "${OUT}/${STRAT}/n${SIZE}/hp${HP_IDX}_${HP_TAG}" \
    --training-sizes "${SIZE}" \
    --chr-split --lr "${LR}" --batch-size "${BS}" \
    --epochs 80 --ensemble-size 1 --early-stop-patience "${PAT}"

echo "=== DONE — $(date) ==="
