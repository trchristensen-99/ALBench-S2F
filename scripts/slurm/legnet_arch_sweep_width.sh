#!/bin/bash
# LegNet width sweep: narrow, default, wide, xwide at large training sizes.
# Tests whether wider models help when more data is available.
#
# Array: one job per width (0=narrow, 1=default, 2=wide, 3=xwide)
#
# Submit:
#   /cm/shared/apps/slurm/current/bin/sbatch --array=0-3 scripts/slurm/legnet_arch_sweep_width.sh
#
#SBATCH --job-name=lgnt_wid
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
CONFIGS=(narrow default wide xwide)
CFG="${CONFIGS[$T]}"

echo "=== LegNet width sweep: ${CFG} node=${SLURMD_NODENAME} $(date) ==="

uv run --no-sync python experiments/legnet_arch_sweep.py \
    --sweep width --config "${CFG}" \
    --sizes 32000 160000 296000 \
    --seeds 3 --lr 0.001 --bs 512 \
    --epochs 80 --patience 10 \
    --output-dir outputs/legnet_arch_sweep

echo "=== Done: ${CFG} $(date) ==="
