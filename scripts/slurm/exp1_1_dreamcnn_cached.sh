#!/bin/bash
# DREAM-CNN student on cached AG S2 pools (supplementary comparison).
# Peter: "include DREAM alongside LegNet and AG"
#
# Uses same cached pools as LegNet — no extra oracle inference needed.
# V100 compatible (no JAX).
#
# Array: one task per Tier 1 strategy
#   0: random  1: genomic  2: prm_5pct  3: evoaug_structural
#
# Submit:
#   /cm/shared/apps/slurm/current/bin/sbatch --array=0-3 scripts/slurm/exp1_1_dreamcnn_cached.sh
#
#SBATCH --job-name=e1_dcnn_pool
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=64G

set -euo pipefail

set +u; source /etc/profile.d/modules.sh; set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
source scripts/slurm/setup_hpc_deps.sh

T=$SLURM_ARRAY_TASK_ID
STRATEGIES=(random genomic prm_5pct evoaug_structural)
STRATEGY="${STRATEGIES[$T]}"
POOL_DIR="outputs/labeled_pools/k562/ag_s2"
OUT_DIR="outputs/exp1_1/k562/dream_cnn_ag_s2"

echo "=== DREAM-CNN + AG S2 pool: ${STRATEGY} node=${SLURMD_NODENAME} $(date) ==="

# Small tier
uv run --no-sync python experiments/exp1_1_scaling.py \
    --task k562 --student dream_cnn --oracle ag_s2 \
    --reservoir "${STRATEGY}" \
    --pool-base-dir "${POOL_DIR}" \
    --n-replicates 3 --seed 42 \
    --output-dir "${OUT_DIR}" \
    --training-sizes 1000 5000 10000 20000 50000 \
    --chr-split \
    --epochs 80 --ensemble-size 1 --early-stop-patience 10 || true

# Large tier
uv run --no-sync python experiments/exp1_1_scaling.py \
    --task k562 --student dream_cnn --oracle ag_s2 \
    --reservoir "${STRATEGY}" \
    --pool-base-dir "${POOL_DIR}" \
    --n-replicates 3 --seed 42 \
    --output-dir "${OUT_DIR}" \
    --training-sizes 100000 200000 500000 \
    --chr-split \
    --epochs 50 --ensemble-size 1 --early-stop-patience 10 \
    --transfer-hp-from 50000 || true

echo "=== Done: $(date) ==="
