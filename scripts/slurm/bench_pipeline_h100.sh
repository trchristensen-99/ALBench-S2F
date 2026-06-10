#!/bin/bash
# Measure the REAL master-pool pipeline rate (gen+label, not pure-predict) on an
# H100 at chunk=128, to compare against the measured V100 rate (~29 seq/s) and size
# an H100-vs-V100 split for the #54 5M relabel.
#   /cm/shared/apps/slurm/current/bin/sbatch scripts/slurm/bench_pipeline_h100.sh
#SBATCH --job-name=bench_pipe_h100
#SBATCH --output=/grid/wsbs/home_norepl/christen/ALBench-S2F/logs/%x-%j.out
#SBATCH --error=/grid/wsbs/home_norepl/christen/ALBench-S2F/logs/%x-%j.err
#SBATCH --partition=gpuq
#SBATCH --qos=fast
#SBATCH --time=00:30:00
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
export AG_ORACLE_CHUNK="${AG_ORACLE_CHUNK:-128}"

ROOT="outputs/_bench_pipe_h100"
rm -rf "${ROOT}"
echo "=== bench_pipe_h100 node=${SLURMD_NODENAME} chunk=${AG_ORACLE_CHUNK} $(date) ==="
uv run --no-sync python scripts/generate_master_pool.py \
    --task k562 --reservoir prm_5pct \
    --target 20000 --n-shards 1 --shard 0 --mode seed --seed 42 --out-root "${ROOT}"
echo "=== BENCH DONE $(date) ==="
