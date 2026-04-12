#!/bin/bash
#SBATCH --job-name=calibrate_oracle_r2
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
source scripts/slurm/setup_hpc_deps.sh
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_command_buffer="
export ALPHAGENOME_WEIGHTS=/grid/wsbs/home_norepl/christen/alphagenome_weights/alphagenome-jax-all_folds-v1

echo "=== calibrate_oracle_r2  node=${SLURMD_NODENAME}  date=$(date) ==="

uv run --no-sync python scripts/calibrate_oracle_r2.py

echo "=== Done $(date) ==="
