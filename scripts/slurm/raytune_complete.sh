#!/bin/bash
# Complete RayTune HP sweep: 6 strategies × 6 sizes = 36 jobs.
# After HP search, runs 3 replicates with best HP.
#
# Strategies: random, genomic, prm_1pct, prm_20pct, motif_grammar, evoaug_prior
# Sizes: 1000, 2000, 5000, 10000, 20000, 50000
#
# Array: strat_idx * 6 + size_idx = 0-35
#
#SBATCH --job-name=rt_comp
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
#SBATCH --partition=gpuq
#SBATCH --time=04:00:00
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

STRATS=("random" "genomic" "prm_1pct" "prm_20pct" "motif_grammar" "evoaug_prior")
SIZES=(1000 2000 5000 10000 20000 50000)

STRAT_IDX=$((T / 6))
SIZE_IDX=$((T % 6))
STRAT=${STRATS[$STRAT_IDX]}
SIZE=${SIZES[$SIZE_IDX]}

echo "=== RayTune + 3 replicates: ${STRAT} n=${SIZE} — $(date) ==="

# Phase 1: RayTune HP search (10 trials)
uv run --no-sync python scripts/raytune_legnet_scaling.py \
    --strategy "${STRAT}" \
    --size "${SIZE}" \
    --seed 42 \
    --n-trials 10 \
    --cpus 8

# Phase 2: Read best HP and run 3 replicates with it
BEST_JSON="outputs/raytune_best/${STRAT}/n${SIZE}/best_config_seed42.json"
if [ -f "${BEST_JSON}" ]; then
    BEST_LR=$(python3 -c "import json; d=json.load(open('${BEST_JSON}')); print(d['config']['lr'])")
    BEST_BS=$(python3 -c "import json; d=json.load(open('${BEST_JSON}')); print(d['config']['batch_size'])")
    echo "Best HP: lr=${BEST_LR}, bs=${BEST_BS}"

    # Determine pool
    POOL_DIR="outputs/labeled_pools_2m/k562/ag_s2"
    [ ! -f "${POOL_DIR}/${STRAT}/pool.npz" ] && POOL_DIR="outputs/labeled_pools/k562/ag_s2"

    OUT="outputs/exp1_1_raytune_final/k562/legnet_ag_s2"

    # Run 3 seeds with best HP
    for SEED in 42 1042 2042; do
        RESULT="${OUT}/${STRAT}/n${SIZE}/hp0/seed${SEED}/result.json"
        [ -f "${RESULT}" ] && echo "SKIP seed=${SEED}" && continue

        echo "--- Replicate seed=${SEED} ---"
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
else
    echo "WARNING: No best HP found, skipping replicates"
fi

echo "=== ALL DONE — $(date) ==="
