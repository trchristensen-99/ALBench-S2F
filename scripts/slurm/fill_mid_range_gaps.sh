#!/bin/bash
# Fill mid-range gaps (100K-500K) for strategies that are missing them.
#
# Gaps:
#   evoaug_heavy:    100K, 200K, 500K (3 sizes × seed 42)
#   motif_grammar:   100K, 200K, 500K (3 sizes × seed 42)
#   motif_planted:   500K (1 size × seed 42)
#   recombination:   500K (1 size × seed 42)
#   gc_matched:      500K (1 size × seed 42)
#
# Array:
#   0-2: evoaug_heavy 100K, 200K, 500K
#   3-5: motif_grammar 100K, 200K, 500K
#   6:   motif_planted 500K
#   7:   recombination_uniform 500K
#   8:   gc_matched 500K
#
# Submit:
#   /cm/shared/apps/slurm/current/bin/sbatch --array=0-8 scripts/slurm/fill_mid_range_gaps.sh
#
#SBATCH --job-name=mid_gaps
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
source scripts/slurm/setup_hpc_deps.sh

T=$SLURM_ARRAY_TASK_ID

STRATS=("evoaug_heavy" "evoaug_heavy" "evoaug_heavy" "motif_grammar" "motif_grammar" "motif_grammar" "motif_planted" "recombination_uniform" "gc_matched")
SIZES=(100000 200000 500000 100000 200000 500000 500000 500000 500000)

STRAT=${STRATS[$T]}
SIZE=${SIZES[$T]}
SEED=42

OUT="outputs/exp1_1/k562/legnet_ag_s2"

[ -f "${OUT}/${STRAT}/n${SIZE}/hp0/seed${SEED}/result.json" ] && echo "SKIP" && exit 0

POOL_DIR="outputs/labeled_pools_2m/k562/ag_s2"

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
