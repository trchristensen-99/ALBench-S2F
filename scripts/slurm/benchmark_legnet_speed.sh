#!/bin/bash
# Benchmark LegNet training speed with various optimization strategies.
#
# Submit:
#   sbatch --qos=fast scripts/slurm/benchmark_legnet_speed.sh
#
#SBATCH --job-name=legnet_speed
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
# Skip setup_hpc_deps.sh — LegNet only needs PyTorch (no AlphaGenome/JAX)

echo "=== legnet_speed_benchmark node=${SLURMD_NODENAME} date=$(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Run all benchmarks with 296K training samples, 10 epochs each
uv run --no-sync python scripts/benchmark_legnet_speed.py \
    --n-train 296000 \
    --epochs 10 \
    --lr 0.005 \
    --output-dir outputs/legnet_speed_benchmark

echo "=== Done $(date) ==="
