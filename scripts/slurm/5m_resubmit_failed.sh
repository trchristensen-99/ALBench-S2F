#!/bin/bash
# Resubmit failed 5M gap-filling jobs with 48h time limit.
#
# Failed jobs: evoaug_heavy s1042/s2042, random s2042
# These timed out at 12h — at 5M sequences, ~640s/epoch, 80 epochs max
# Need ~14-16h for training + eval.
#
# Array:
#   0: evoaug_heavy seed=1042
#   1: evoaug_heavy seed=2042
#   2: random seed=2042
#
#SBATCH --job-name=5m_resub
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

STRATS=("evoaug_heavy" "evoaug_heavy" "random")
SEEDS=(1042 2042 2042)

STRAT=${STRATS[$T]}
SEED=${SEEDS[$T]}

OUT="outputs/exp1_1_5m_scaling/k562/legnet_ag_s2"

[ -f "${OUT}/${STRAT}/n5000000/hp0/seed${SEED}/result.json" ] && echo "SKIP" && exit 0

echo "=== 5M ${STRAT} seed=${SEED} — $(date) ==="

uv run --no-sync python experiments/exp1_1_scaling.py \
    --task k562 --student legnet --oracle ag_s2 \
    --reservoir "${STRAT}" \
    --pool-base-dir outputs/labeled_pools_2m/k562/ag_s2 \
    --n-replicates 1 --seed "${SEED}" \
    --output-dir "${OUT}" \
    --training-sizes 5000000 \
    --chr-split --lr 0.005 --batch-size 1024 \
    --epochs 80 --ensemble-size 1 --early-stop-patience 10

echo "=== DONE — $(date) ==="
