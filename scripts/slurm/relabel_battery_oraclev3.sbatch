#!/bin/bash
# Relabel the chr-split eval battery uniformly with the oracle_v3 ensemble.
# Submit: /cm/shared/apps/slurm/current/bin/sbatch scripts/slurm/relabel_battery_oraclev3.sbatch
#SBATCH --job-name=relabel_battery_v3
#SBATCH --partition=gpuq
#SBATCH --qos=fast
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=4:00:00
#SBATCH --requeue
#SBATCH --output=/grid/wsbs/home_norepl/christen/ALBench-S2F/outputs/relabel_battery/logs/%x_%j.log

set -euo pipefail
set +u; source /etc/profile.d/modules.sh; [ -f ~/.bash_profile ] && source ~/.bash_profile; set -u
module load EB5 2>/dev/null || true
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
mkdir -p outputs/relabel_battery/logs
source .venv/bin/activate 2>/dev/null || true
export PYTHONPATH="$PWD:$PWD/experiments${PYTHONPATH:+:$PYTHONPATH}"
source scripts/slurm/setup_hpc_deps.sh
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_command_buffer="
export TQDM_DISABLE=1
export PYTHONUNBUFFERED=1
export AG_ORACLE_CHUNK="${AG_ORACLE_CHUNK:-128}"

echo "=== relabel_battery_v3 node=${SLURMD_NODENAME:-?} $(date) ==="
uv run --no-sync python scripts/relabel_battery_oraclev3_ensemble.py
echo "=== relabel_battery_v3 DONE $(date) ==="
