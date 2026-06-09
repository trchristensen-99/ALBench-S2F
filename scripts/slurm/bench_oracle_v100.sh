#!/bin/bash
# V100 throughput probe for the #54 master-pool relabel. Labeling is GPU-invariant
# (only the student-training HP sweep needs GPU-seconds comparability), so the relabel
# can run on idle V100s — but the 551M-param backbone OOMs at the H100 chunk of 128 on
# a V100 32GB. This measures seq/s at AG_ORACLE_CHUNK=32 to size the array walltime.
#   /cm/shared/apps/slurm/current/bin/sbatch scripts/slurm/bench_oracle_v100.sh
#SBATCH --job-name=bench_oracle_v100
#SBATCH --output=/grid/wsbs/home_norepl/christen/ALBench-S2F/logs/%x-%j.out
#SBATCH --error=/grid/wsbs/home_norepl/christen/ALBench-S2F/logs/%x-%j.err
#SBATCH --partition=gpuq
#SBATCH --qos=fast
#SBATCH --time=00:40:00
#SBATCH --gres=gpu:v100:1
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
export AG_ORACLE_CHUNK="${AG_ORACLE_CHUNK:-32}"

echo "=== bench_oracle_v100 chunk=${AG_ORACLE_CHUNK} node=${SLURMD_NODENAME} $(date) ==="
uv run --no-sync python scripts/debug/bench_oracle_throughput.py
echo "=== BENCH DONE ==="
