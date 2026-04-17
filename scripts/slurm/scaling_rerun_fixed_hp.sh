#!/bin/bash
# Full scaling rerun with size-calibrated fixed HPs.
#
# Based on pilot results + HP search:
#   1K-2K:  lr=0.0005, bs=128 (small N needs low LR, small BS)
#   5K-10K: lr=0.003, bs=256 (pilot optimal)
#   20K+:   lr=0.003, bs=256 (pilot optimal; also test lr=0.005/bs=512)
#
# 6 strategies × 8 sizes × 3 seeds = 144 jobs
# Plus 6 strategies × 3 large sizes × 3 seeds with alt HP = 54 more = 198 total
#
# Array: config * 3 + seed_idx
# configs 0-47: main HP (lr=0.003/bs=256 or lr=0.0005/bs=128 for small N)
# configs 48-65: alt HP at 20K/50K/100K (lr=0.005/bs=512)
#
#SBATCH --job-name=scl_rr
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

STRATS=("random" "genomic" "prm_1pct" "prm_20pct" "motif_grammar" "evoaug_heavy")
SEEDS=(42 1042 2042)

SEED_IDX=$((T % 3))
CONFIG_IDX=$((T / 3))
SEED=${SEEDS[$SEED_IDX]}

# Main HP configs: 6 strats × 8 sizes = 48 configs (T/3 = 0-47)
# Alt HP configs: 6 strats × 3 sizes = 18 configs (T/3 = 48-65)
if [ $CONFIG_IDX -lt 48 ]; then
    STRAT_IDX=$((CONFIG_IDX / 8))
    SIZE_IDX=$((CONFIG_IDX % 8))
    SIZES=(1000 2000 5000 10000 20000 50000 100000 200000)
    SIZE=${SIZES[$SIZE_IDX]}
    STRAT=${STRATS[$STRAT_IDX]}
    
    # Size-calibrated HP
    if [ $SIZE -le 2000 ]; then
        LR=0.0005
        BS=128
        HP_TAG="hp_small"
    else
        LR=0.003
        BS=256
        HP_TAG="hp_main"
    fi
else
    # Alt HP: lr=0.005, bs=512 at 20K/50K/100K
    ALT_IDX=$((CONFIG_IDX - 48))
    STRAT_IDX=$((ALT_IDX / 3))
    SIZE_IDX=$((ALT_IDX % 3))
    ALT_SIZES=(20000 50000 100000)
    SIZE=${ALT_SIZES[$SIZE_IDX]}
    STRAT=${STRATS[$STRAT_IDX]}
    LR=0.005
    BS=512
    HP_TAG="hp_alt"
fi

POOL_DIR="outputs/labeled_pools/k562/ag_s2"
OUT="outputs/exp1_1_rerun/k562/legnet_ag_s2"
RESULT="${OUT}/${STRAT}/n${SIZE}/${HP_TAG}/seed${SEED}/result.json"

[ -f "${RESULT}" ] && echo "SKIP" && exit 0

# Stagger to avoid NFS issues
sleep $((RANDOM % 5))

echo "=== Rerun: ${STRAT} n=${SIZE} seed=${SEED} lr=${LR} bs=${BS} — $(date) ==="

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
