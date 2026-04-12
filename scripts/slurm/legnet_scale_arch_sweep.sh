#!/bin/bash
# LegNet architecture sweep at large N (1M examples from oracle pool).
#
# Tests 5 configs: default, wide, wide_ks7, depth10_wide, default_ks7
# at n=1M and n=296K (for comparison with existing sweep data).
#
# Each SLURM array task handles one config × one size × seed=42.
# Array mapping (10 tasks):
#   0 : default       n=296K
#   1 : wide          n=296K
#   2 : wide_ks7      n=296K
#   3 : depth10_wide  n=296K
#   4 : default_ks7   n=296K
#   5 : default       n=1M
#   6 : wide          n=1M
#   7 : wide_ks7      n=1M
#   8 : depth10_wide  n=1M
#   9 : default_ks7   n=1M
#
# Submit all:
#   /cm/shared/apps/slurm/current/bin/sbatch --array=0-9 scripts/slurm/legnet_scale_arch_sweep.sh
#
# Submit only 1M (tasks 5-9):
#   /cm/shared/apps/slurm/current/bin/sbatch --array=5-9 scripts/slurm/legnet_scale_arch_sweep.sh
#
#SBATCH --job-name=lgnt_scale
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=default
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G

set -euo pipefail

set +u; source /etc/profile.d/modules.sh; set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
source scripts/slurm/setup_hpc_deps.sh
export TORCHDYNAMO_DISABLE=1
export TORCH_COMPILE_DISABLE=1

T=$SLURM_ARRAY_TASK_ID

# Config and size arrays (must stay in sync with array mapping above)
CONFIGS=(default wide wide_ks7 depth10_wide default_ks7 default wide wide_ks7 depth10_wide default_ks7)
SIZES=(  296000  296000  296000      296000       296000 1000000 1000000 1000000     1000000      1000000)

CFG="${CONFIGS[$T]}"
N="${SIZES[$T]}"

echo "=== LegNet scale arch sweep: cfg=${CFG} n=${N} task=${T} node=${SLURMD_NODENAME} $(date) ==="

uv run --no-sync python experiments/legnet_scale_arch_sweep.py \
    --config "${CFG}" \
    --sizes "${N}" \
    --seeds 1 \
    --lr 0.001 --bs 512 \
    --patience 8 \
    --output-dir outputs/legnet_arch_sweep

echo "=== Done: cfg=${CFG} n=${N} $(date) ==="
