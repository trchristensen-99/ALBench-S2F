#!/bin/bash
# LegNet depth sweep: 4, 6, 8, 10 blocks at multiple training sizes.
# Tests whether deeper models help for MPRA sequence prediction.
#
# Array: one job per depth (0=depth4, 1=depth6, 2=depth8, 3=depth10)
#
# Submit:
#   /cm/shared/apps/slurm/current/bin/sbatch --array=0-3 scripts/slurm/legnet_arch_sweep_depth.sh
#
#SBATCH --job-name=lgnt_dep
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=default
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

set -euo pipefail

set +u; source /etc/profile.d/modules.sh; set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
source scripts/slurm/setup_hpc_deps.sh
export TORCHDYNAMO_DISABLE=1
export TORCH_COMPILE_DISABLE=1

T=$SLURM_ARRAY_TASK_ID
CONFIGS=(depth4 depth6 depth8 depth10)
CFG="${CONFIGS[$T]}"

echo "=== LegNet depth sweep: ${CFG} node=${SLURMD_NODENAME} $(date) ==="

uv run --no-sync python experiments/legnet_arch_sweep.py \
    --sweep depth --config "${CFG}" \
    --sizes 32000 160000 296000 \
    --seeds 3 --lr 0.001 --bs 512 \
    --epochs 80 --patience 10 \
    --output-dir outputs/legnet_arch_sweep

echo "=== Done: ${CFG} $(date) ==="
