#!/bin/bash
# FINAL 6-strategy scaling curves for Peter's talk.
#
# 6 strategies × 5 sizes × 4 HPs × 3 seeds = 360 jobs
# BUT we use exp1_1_scaling.py which handles multiple HPs and seeds per job
# So: 6 strategies × 5 sizes × 1 job = 30 jobs, each doing 4 HPs × 3 seeds internally
#
# Strategies: random, genomic, prm_1pct, prm_20pct, motif_grammar, evoaug_heavy
# Sizes: 1000, 5000, 10000, 20000, 50000
# HPs: 4 configs (bs=512/1024 × lr=0.001/0.005)
# Seeds: 42, 1042, 2042
#
# Submit all on FAST queue for priority:
#   /cm/shared/apps/slurm/current/bin/sbatch --array=0-29 scripts/slurm/peter6_final.sh
#
#SBATCH --job-name=p6_fin
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

STRATS=("random" "genomic" "prm_1pct" "prm_20pct" "motif_grammar" "evoaug_heavy")
SIZES=(1000 5000 10000 20000 50000)

STRAT_IDX=$((T / 5))
SIZE_IDX=$((T % 5))
STRAT=${STRATS[$STRAT_IDX]}
SIZE=${SIZES[$SIZE_IDX]}

OUT="outputs/exp1_1_peter6_final/k562/legnet_ag_s2"

POOL_DIR="outputs/labeled_pools_2m/k562/ag_s2"
[ ! -f "${POOL_DIR}/${STRAT}/pool.npz" ] && POOL_DIR="outputs/labeled_pools/k562/ag_s2"

echo "=== ${STRAT} n=${SIZE} — $(date) ==="

# Run 3 seeds with 4 HP configs each (exp1_1_scaling handles this internally)
uv run --no-sync python experiments/exp1_1_scaling.py \
    --task k562 --student legnet --oracle ag_s2 \
    --reservoir "${STRAT}" \
    --pool-base-dir "${POOL_DIR}" \
    --n-replicates 3 --seed 42 \
    --output-dir "${OUT}" \
    --training-sizes "${SIZE}" \
    --chr-split \
    --epochs 80 --ensemble-size 1 --early-stop-patience 10

echo "=== DONE — $(date) ==="
