#!/bin/bash
# LegNet with finer training size steps on key strategies.
# Peter: "It's like learning rates: if the rate is small, you'll have to
# do more experiments. Trying to establish what the scaling law curve looks like."
#
# Current sizes: 1K, 5K, 10K, 20K, 50K, 100K, 200K, 500K
# Fine sizes add: 2K, 3K, 7K, 15K, 30K, 75K, 150K, 300K
# This doubles the resolution on the scaling curve.
#
# Only runs on genomic and random (most important for establishing the curve shape).
# Uses no HP sweep (best HP from main runs) for speed.
#
# Array: 0=genomic, 1=random
#
# Submit:
#   /cm/shared/apps/slurm/current/bin/sbatch --array=0-1 scripts/slurm/exp1_1_legnet_fine_sizes.sh
#
#SBATCH --job-name=lgnt_fine
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
STRATEGIES=(genomic random)
STRATEGY="${STRATEGIES[$T]}"
POOL_DIR="outputs/labeled_pools/k562/ag_s2"
OUT_DIR="outputs/exp1_1/k562/legnet_ag_s2"

echo "=== LegNet fine sizes: ${STRATEGY} node=${SLURMD_NODENAME} $(date) ==="

# Use best HP from main runs (lr=0.001, bs=1024)
uv run --no-sync python experiments/exp1_1_scaling.py \
    --task k562 --student legnet --oracle ag_s2 \
    --reservoir "${STRATEGY}" \
    --pool-base-dir "${POOL_DIR}" \
    --n-replicates 3 --seed 42 \
    --output-dir "${OUT_DIR}" \
    --training-sizes 2000 3000 7000 15000 30000 75000 150000 300000 \
    --lr 0.001 --batch-size 1024 \
    --chr-split \
    --epochs 80 --ensemble-size 1 --early-stop-patience 10 \
    --save-predictions

echo "=== Done: $(date) ==="
