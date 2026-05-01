#!/bin/bash
# DREAM-CNN yeast — 2 more seeds (3042, 4042) at full data on real labels.
# Combined with the existing 3 seeds (42, 1042, 2042), gives a 5-DCNN
# ensemble that pairs with the existing 5-DRNN ensemble for the
# 5+5 mixture oracle pseudolabel generation.
#
# Submit:
#   /cm/shared/apps/slurm/current/bin/sbatch scripts/slurm/yeast_dcnn_real_extra_seeds.sh
#
#SBATCH --job-name=yeast_dcnn_real_x
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --array=0-1

set -euo pipefail
set +u; source /etc/profile.d/modules.sh; set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD"
export TORCHDYNAMO_DISABLE=1
source scripts/slurm/setup_hpc_deps.sh

N=6065324
SEEDS=(3042 4042)
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
