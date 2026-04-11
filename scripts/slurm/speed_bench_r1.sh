#!/bin/bash
# Round 1 LegNet speed benchmark — H100, fast QoS (4h).
#
# Submit:
#   sbatch scripts/slurm/speed_bench_r1.sh
#
#SBATCH --job-name=speed_r1
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

echo "=== speed_bench_r1 node=${SLURMD_NODENAME} date=$(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

uv run --no-sync python scripts/speed_bench_r1.py \
    --data-path outputs/labeled_pools/k562/ag_s2/genomic/pool.npz \
    --n-train 296000 \
    --epochs 12 \
    --output-dir outputs/legnet_speed_r1

echo "=== Done $(date) ==="
