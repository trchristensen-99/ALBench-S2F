#!/bin/bash
# Extra replicates at small N (1K-5K) for tighter CIs.
#
# Uses the best HP from 20-trial warm-start Optuna (seed 42).
# Runs 10 additional seeds (3042-12042) per (strategy, size).
# 6 strategies × 3 sizes × 10 seeds = 180 jobs
#
# Array: strat_idx * 30 + size_idx * 10 + seed_idx
# But that's too many — instead run 1 job per (strategy, size)
# that does 10 seeds internally.
#
# 6 strategies × 3 sizes = 18 jobs
# Array: strat_idx * 3 + size_idx
#
#SBATCH --job-name=xtra_rp
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
export TORCHDYNAMO_DISABLE=1
source scripts/slurm/setup_hpc_deps.sh

T=$SLURM_ARRAY_TASK_ID

STRATS=("random" "genomic" "prm_1pct" "prm_20pct" "motif_grammar" "evoaug_heavy")
SIZES=(1000 2000 5000)

STRAT_IDX=$((T / 3))
SIZE_IDX=$((T % 3))
STRAT=${STRATS[$STRAT_IDX]}
SIZE=${SIZES[$SIZE_IDX]}

echo "=== Extra replicates: ${STRAT} n=${SIZE} — $(date) ==="

# Read best HP from optuna_best (20-trial warm-start)
BEST_JSON="outputs/optuna_best/${STRAT}/n${SIZE}/best_config_seed42.json"
if [ -f "${BEST_JSON}" ]; then
    BEST_LR=$(python3 -c "import json; print(json.load(open('${BEST_JSON}'))['config']['lr'])")
    BEST_BS=$(python3 -c "import json; print(json.load(open('${BEST_JSON}'))['config']['batch_size'])")
else
    # Fallback to known-good defaults
    BEST_LR=0.002
    BEST_BS=256
fi

echo "  Using HP: lr=${BEST_LR} bs=${BEST_BS}"

POOL_DIR="outputs/labeled_pools_2m/k562/ag_s2"
[ ! -f "${POOL_DIR}/${STRAT}/pool.npz" ] && POOL_DIR="outputs/labeled_pools/k562/ag_s2"

OUT="outputs/exp1_1_extra_replicates/k562/legnet_ag_s2"

# Run 10 extra seeds
for SEED in 3042 4042 5042 6042 7042 8042 9042 10042 11042 12042; do
    [ -f "${OUT}/${STRAT}/n${SIZE}/hp0/seed${SEED}/result.json" ] && echo "  SKIP seed=${SEED}" && continue
    echo "  Running seed=${SEED}..."
    uv run --no-sync python experiments/exp1_1_scaling.py \
        --task k562 --student legnet --oracle ag_s2 \
        --reservoir "${STRAT}" \
        --pool-base-dir "${POOL_DIR}" \
        --n-replicates 1 --seed "${SEED}" \
        --output-dir "${OUT}" \
        --training-sizes "${SIZE}" \
        --chr-split --lr "${BEST_LR}" --batch-size "${BEST_BS}" \
        --epochs 80 --ensemble-size 1 --early-stop-patience 10
done

echo "=== DONE — $(date) ==="
