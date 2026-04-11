#!/bin/bash
# LegNet kernel size sweep: ks=3, 5, 7, 9 at multiple training sizes.
# Tests whether larger receptive fields help for MPRA sequence prediction.
#
# Array: one job per kernel size (0=ks3, 1=ks5, 2=ks7, 3=ks9)
#
# Submit:
#   /cm/shared/apps/slurm/current/bin/sbatch --array=0-3 scripts/slurm/legnet_arch_sweep_kernel.sh
#
#SBATCH --job-name=lgnt_ks
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
CONFIGS=(ks3 ks5 ks7 ks9)
CFG="${CONFIGS[$T]}"

echo "=== LegNet kernel sweep: ${CFG} node=${SLURMD_NODENAME} $(date) ==="

uv run --no-sync python experiments/legnet_arch_sweep.py \
    --sweep kernel --config "${CFG}" \
    --sizes 32000 160000 296000 \
    --seeds 3 --lr 0.001 --bs 512 \
    --epochs 80 --patience 10 \
    --output-dir outputs/legnet_arch_sweep

echo "=== Done: ${CFG} $(date) ==="
