#!/bin/bash
# LR schedule benchmark for LegNet — convergence speed comparison.
#
# Tests 4 schedules at n=296K and reports val Pearson r at ep 10/20/30/40.
# Round 1: 5 initial schedules × 2 seeds each  (~3-4h on H100)
# Round 2: 6 refined schedules × 3 seeds each  (~4-5h on H100)
#
# Submit Round 1:
#   /cm/shared/apps/slurm/current/bin/sbatch \
#     --export=ROUND=1 \
#     scripts/slurm/lr_schedule_benchmark.sh
#
# Submit Round 2 (after reviewing Round 1 results):
#   /cm/shared/apps/slurm/current/bin/sbatch \
#     --export=ROUND=2 \
#     scripts/slurm/lr_schedule_benchmark.sh
#
#SBATCH --job-name=legnet_lr_bench
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpuq
#SBATCH --qos=fast
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G

set -euo pipefail

set +u; source /etc/profile.d/modules.sh; set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

# Skip setup_hpc_deps.sh — LegNet only needs PyTorch
echo "=== legnet_lr_bench node=${SLURMD_NODENAME} date=$(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

ROUND=${ROUND:-1}
echo "Running Round ${ROUND}"

uv run --no-sync python scripts/lr_schedule_benchmark.py \
    --data-path outputs/labeled_pools/k562/ag_s2/genomic/pool.npz \
    --n-train 296382 \
    --round "${ROUND}" \
    --output-dir "outputs/lr_schedule_benchmark"

echo "=== Done $(date) ==="
