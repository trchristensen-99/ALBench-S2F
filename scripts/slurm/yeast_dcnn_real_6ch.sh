#!/bin/bash
# DREAM-CNN yeast — 6-channel input (Prix Fixe / pilot-paper spec):
# 4 nucleotides + RC orientation flag + singleton flag.
# Compares against existing 4-channel DCNN (in_dist_real ~0.806, ood_real ~0.652).
#
# 3 seeds at full data so we can compare distributions.
#
#SBATCH --job-name=yeast_dcnn_6ch
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --array=0-2

set -euo pipefail
set +u; source /etc/profile.d/modules.sh; set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD"
export TORCHDYNAMO_DISABLE=1
source scripts/slurm/setup_hpc_deps.sh

N=6065324
SEEDS=(42 1042 2042)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

# DREAMCNNStudent now defaults to 6 channels for task_mode='yeast' (was 4).
# This matches the DREAM Challenge / Prix-Fixe / pilot-paper spec.
uv run --no-sync python experiments/exp1_1_scaling.py \
    --task yeast \
    --student dream_cnn \
    --oracle ground_truth --reservoir genomic \
    --n-replicates 1 --no-hp-sweep --seed "${SEED}" \
    --lr 0.005 --batch-size 1024 \
    --output-dir "outputs/exp0_yeast_dream_cnn_real_6ch" \
    --training-sizes "${N}" \
    --epochs 80 \
    --ensemble-size 1 \
    --early-stop-patience 10

echo "=== DONE — $(date) ==="
