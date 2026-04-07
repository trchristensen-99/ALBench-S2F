#!/bin/bash
# Generate predictions.npz for LegNet in chr_split (all 3 cells).
# LegNet checkpoints exist but no predictions were saved during training.
#
# Submit:
#   /cm/shared/apps/slurm/current/bin/sbatch --array=0-2 scripts/slurm/save_legnet_chr_split_predictions.sh
#
#SBATCH --job-name=lgnt_pred
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=default
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

set -euo pipefail

set +u; source /etc/profile.d/modules.sh; set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
source scripts/slurm/setup_hpc_deps.sh

T=$SLURM_ARRAY_TASK_ID
CELLS=("k562" "hepg2" "sknsh")
CELL="${CELLS[$T]}"

echo "=== LegNet predictions for ${CELL} — $(date) ==="

uv run --no-sync python scripts/save_from_scratch_chr_split_predictions.py \
    --cell "${CELL}" \
    --model legnet \
    --force

echo "=== Done: $(date) ==="
