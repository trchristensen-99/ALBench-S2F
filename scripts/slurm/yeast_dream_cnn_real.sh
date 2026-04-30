#!/bin/bash
# DREAM-CNN yeast — real-label training at full data, 3 seeds.
# Uses --reservoir genomic --oracle ground_truth (the only combo allowed for
# real labels in exp1_1_scaling.py — yeast has no synthetic-random ground_truth path).
# This provides the apples-to-apples oracle comparison vs DRNN and AG S2 (real).
#
# Submit:
#   /cm/shared/apps/slurm/current/bin/sbatch scripts/slurm/yeast_dream_cnn_real.sh
#
#SBATCH --job-name=yeast_dcnn_real
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

uv run --no-sync python experiments/exp1_1_scaling.py \
    --task yeast \
    --student dream_cnn \
    --oracle ground_truth --reservoir genomic \
    --n-replicates 1 --no-hp-sweep --seed "${SEED}" \
    --lr 0.005 --batch-size 1024 \
    --output-dir "outputs/exp0_yeast_dream_cnn_real" \
    --training-sizes "${N}" \
    --epochs 80 \
    --ensemble-size 1 \
    --early-stop-patience 10

echo "=== DONE — $(date) ==="
