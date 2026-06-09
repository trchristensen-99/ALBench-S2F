#!/bin/bash
# Measure AG_S2 oracle labeling throughput on an H100 (sizing for #54 relabel).
#   export PATH=/cm/shared/apps/slurm/current/bin:$PATH
#   sbatch scripts/slurm/bench_oracle.sh
#SBATCH --job-name=bench_oracle
#SBATCH --output=/grid/wsbs/home_norepl/christen/ALBench-S2F/logs/%x-%j.out
#SBATCH --error=/grid/wsbs/home_norepl/christen/ALBench-S2F/logs/%x-%j.err
#SBATCH --partition=gpuq
#SBATCH --qos=default
#SBATCH --time=00:40:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G

set -euo pipefail
set +u; source /etc/profile.d/modules.sh; set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
source scripts/slurm/setup_hpc_deps.sh
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_command_buffer="
export TQDM_DISABLE=1
export PYTHONUNBUFFERED=1

uv run --no-sync python scripts/debug/bench_oracle_throughput.py
echo "=== BENCH DONE ==="
