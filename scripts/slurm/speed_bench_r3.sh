#!/bin/bash
# Round 3 LegNet speed benchmark — multi-seed quality + speed validation.
# Submit AFTER reviewing R1+R2 results.
#
# Submit:
#   sbatch scripts/slurm/speed_bench_r3.sh
#   # Or override which configs to run:
#   CONFIGS="production_baseline best_combined" sbatch scripts/slurm/speed_bench_r3.sh
#
#SBATCH --job-name=speed_r3
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpuq
#SBATCH --qos=fast
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G

set -euo pipefail
set +u; source /etc/profile.d/modules.sh; set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

BEST_CONFIG="${BEST_CONFIG:-best_combined}"
CONFIGS_ARG="${CONFIGS:-production_baseline best_combined}"

echo "=== speed_bench_r3 node=${SLURMD_NODENAME} date=$(date) ==="
echo "=== BEST_CONFIG=${BEST_CONFIG} CONFIGS=${CONFIGS_ARG} ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

uv run --no-sync python scripts/speed_bench_r3.py \
    --data-path outputs/labeled_pools/k562/ag_s2/genomic/pool.npz \
    --n-train 296000 \
    --best-config "${BEST_CONFIG}" \
    --configs ${CONFIGS_ARG} \
    --output-dir outputs/legnet_speed_r3

echo "=== Done $(date) ==="
