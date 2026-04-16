#!/bin/bash
# Redo LegNet + AG S2 oracle at AG-aligned sizes with fixed known-good HP.
#
# Uses the genomic pool (618K seqs with AG S2 oracle labels).
# Sizes match AG ground truth: 3197, 6395, 15987, 31974, 63949, 159871, 296382
# 3 replicates each = 21 jobs
# Fixed HP: lr=0.003, bs=256 (from HP comparison)
#
# Array: size_idx * 3 + rep_idx
#
#SBATCH --job-name=exp0_ln
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
export TORCHDYNAMO_DISABLE=1
source scripts/slurm/setup_hpc_deps.sh

T=$SLURM_ARRAY_TASK_ID

SIZES=(3197 6395 15987 31974 63949 159871 296382)
SEEDS=(42 1042 2042)

SIZE_IDX=$((T / 3))
REP_IDX=$((T % 3))
SIZE=${SIZES[$SIZE_IDX]}
SEED=${SEEDS[$REP_IDX]}

POOL_DIR="outputs/labeled_pools/k562/ag_s2"
OUT="outputs/exp0_legnet_ag_s2_redo/k562/legnet_ag_s2"
RESULT="${OUT}/genomic/n${SIZE}/hp0/seed${SEED}/result.json"

[ -f "${RESULT}" ] && echo "SKIP" && exit 0

echo "=== Exp0 LegNet redo: n=${SIZE} seed=${SEED} — $(date) ==="

uv run --no-sync python experiments/exp1_1_scaling.py \
    --task k562 --student legnet --oracle ag_s2 \
    --reservoir genomic \
    --pool-base-dir "${POOL_DIR}" \
    --n-replicates 1 --seed "${SEED}" \
    --output-dir "${OUT}" \
    --training-sizes "${SIZE}" \
    --chr-split --lr 0.003 --batch-size 256 \
    --epochs 80 --ensemble-size 1 --early-stop-patience 10

echo "=== DONE — $(date) ==="
