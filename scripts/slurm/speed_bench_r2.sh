#!/bin/bash
# Round 2 LegNet speed benchmark — fused AdamW, channels_last, fullgraph, bf16 combos.
# Submit AFTER reviewing R1 results.
#
# Submit:
#   sbatch scripts/slurm/speed_bench_r2.sh
#
#SBATCH --job-name=speed_r2
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

# Pass R1 winner name here for logging
R1_WINNER="${R1_WINNER:-bf16_compile}"

echo "=== speed_bench_r2 node=${SLURMD_NODENAME} date=$(date) ==="
echo "=== R1_WINNER=${R1_WINNER} ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

uv run --no-sync python scripts/speed_bench_r2.py \
    --data-path outputs/labeled_pools/k562/ag_s2/genomic/pool.npz \
    --n-train 296000 \
    --epochs 12 \
    --winner "${R1_WINNER}" \
    --output-dir outputs/legnet_speed_r2

echo "=== Done $(date) ==="
