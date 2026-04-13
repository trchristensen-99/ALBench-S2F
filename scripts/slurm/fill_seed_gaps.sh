#!/bin/bash
# Fill ALL missing seeds to get ≥3 seeds per (strategy, size) for CIs.
#
# Generated from gap analysis. Each task runs one (strategy, size, seed).
#
#SBATCH --job-name=seed_gap
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

# Each row: strategy, size, seed
# Ordered by priority (most visible gaps first)
read STRAT SIZE SEED << EOF
$(sed -n "$((T+1))p" << 'GAPS'
random 1000000 42
random 2000000 1042
random 5000000 2042
prm_5pct 1000000 42
prm_5pct 2000000 1042
prm_5pct 5000000 2042
evoaug_structural 2000000 1042
evoaug_structural 5000000 42
evoaug_structural 5000000 2042
dinuc_shuffle 1000000 1042
dinuc_shuffle 1000000 2042
dinuc_shuffle 2000000 1042
dinuc_shuffle 2000000 2042
motif_grammar 1000000 1042
motif_grammar 1000000 2042
motif_grammar 2000000 1042
motif_planted 200000 2042
motif_planted 1000000 1042
motif_planted 1000000 2042
motif_planted 2000000 1042
recombination_uniform 1000000 1042
recombination_uniform 1000000 2042
recombination_uniform 2000000 1042
evoaug_heavy 1000000 1042
evoaug_heavy 1000000 2042
evoaug_heavy 2000000 1042
evoaug_heavy 5000000 1042
evoaug_heavy 5000000 2042
GAPS
)
EOF

if [ -z "$STRAT" ]; then
    echo "Task $T: no gap defined"
    exit 0
fi

# Determine output dir based on size
if [ "$SIZE" -ge 5000000 ]; then
    OUT="outputs/exp1_1_5m_scaling/k562/legnet_ag_s2"
elif [ "$SIZE" -ge 1000000 ]; then
    OUT="outputs/exp1_1_2m_scaling/k562/legnet_ag_s2"
else
    OUT="outputs/exp1_1/k562/legnet_ag_s2"
fi

[ -f "${OUT}/${STRAT}/n${SIZE}/hp0/seed${SEED}/result.json" ] && echo "SKIP: ${STRAT} n=${SIZE} seed=${SEED}" && exit 0

POOL_DIR="outputs/labeled_pools_2m/k562/ag_s2"
# Fall back to original pool if 2m doesn't exist
[ ! -f "${POOL_DIR}/${STRAT}/pool.npz" ] && POOL_DIR="outputs/labeled_pools/k562/ag_s2"

echo "=== ${STRAT} n=${SIZE} seed=${SEED} — $(date) ==="

uv run --no-sync python experiments/exp1_1_scaling.py \
    --task k562 --student legnet --oracle ag_s2 \
    --reservoir "${STRAT}" \
    --pool-base-dir "${POOL_DIR}" \
    --n-replicates 1 --seed "${SEED}" \
    --output-dir "${OUT}" \
    --training-sizes "${SIZE}" \
    --chr-split --lr 0.005 --batch-size 1024 \
    --epochs 80 --ensemble-size 1 --early-stop-patience 10

echo "=== DONE — $(date) ==="
