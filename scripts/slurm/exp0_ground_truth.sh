#!/bin/bash
# Exp0 ground truth label scaling for all from-scratch models.
# Uses --oracle ground_truth --reservoir genomic.
#
# Array:
#   0: DREAM-CNN K562 ground truth
#   1: DREAM-RNN K562 ground truth
#   2: LegNet K562 ground truth (already partial, will skip completed)
#   3: DREAM-CNN Yeast ground truth
#   4: DREAM-RNN Yeast ground truth
#   5: LegNet Yeast ground truth (already partial, will skip completed)
#
# AG S1/S2 ground truth not included since AG uses cached embeddings
# which require a different pipeline (train_foundation_cached.py).
#
# Submit:
#   sbatch --array=0-2 --qos=default --time=12:00:00 scripts/slurm/exp0_ground_truth.sh
#   sbatch --array=3-5 --qos=slow_nice scripts/slurm/exp0_ground_truth.sh
#
#SBATCH --job-name=exp0_gt
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=200G

set -euo pipefail

set +u; source /etc/profile.d/modules.sh; set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
source scripts/slurm/setup_hpc_deps.sh

T=$SLURM_ARRAY_TASK_ID
echo "=== exp0_ground_truth task=${T} node=${SLURMD_NODENAME} date=$(date) ==="

K562_SIZES="3197 6395 15987 31974 63949 159871 319742"
YEAST_SIZES="6065 12131 30327 60653 121307 303266 606532 1213065 3032662 6065324"

case ${T} in
0)  echo "DREAM-CNN K562 ground truth"
    uv run --no-sync python experiments/exp1_1_scaling.py \
        --task k562 --student dream_cnn \
        --oracle ground_truth --reservoir genomic \
        --n-replicates 3 --seed 42 \
        --output-dir "outputs/exp0_oracle_scaling_v4/k562/dream_cnn_ground_truth" \
        --training-sizes ${K562_SIZES} --epochs 80 \
        --early-stop-patience 10 --ensemble-size 1
    ;;
1)  echo "DREAM-RNN K562 ground truth"
    uv run --no-sync python experiments/exp1_1_scaling.py \
        --task k562 --student dream_rnn \
        --oracle ground_truth --reservoir genomic \
        --n-replicates 3 --seed 42 \
        --output-dir "outputs/exp0_oracle_scaling_v4/k562/dream_rnn_ground_truth" \
        --training-sizes ${K562_SIZES} --epochs 80 \
        --early-stop-patience 10 --ensemble-size 1
    ;;
2)  echo "LegNet K562 ground truth (resume)"
    uv run --no-sync python experiments/exp1_1_scaling.py \
        --task k562 --student legnet \
        --oracle ground_truth --reservoir genomic \
        --n-replicates 3 --seed 42 \
        --output-dir "outputs/exp0_oracle_scaling_v4/k562/legnet_ground_truth" \
        --training-sizes ${K562_SIZES} --epochs 80 \
        --early-stop-patience 10 --ensemble-size 1
    ;;
3)  echo "DREAM-CNN Yeast ground truth"
    uv run --no-sync python experiments/exp1_1_scaling.py \
        --task yeast --student dream_cnn \
        --oracle ground_truth --reservoir genomic \
        --n-replicates 3 --seed 42 \
        --output-dir "outputs/exp0_oracle_scaling_v4/yeast/dream_cnn_ground_truth" \
        --training-sizes ${YEAST_SIZES} --epochs 80 \
        --early-stop-patience 10 --ensemble-size 1
    ;;
4)  echo "DREAM-RNN Yeast ground truth"
    uv run --no-sync python experiments/exp1_1_scaling.py \
        --task yeast --student dream_rnn \
        --oracle ground_truth --reservoir genomic \
        --n-replicates 3 --seed 42 \
        --output-dir "outputs/exp0_oracle_scaling_v4/yeast/dream_rnn_ground_truth" \
        --training-sizes ${YEAST_SIZES} --epochs 80 \
        --early-stop-patience 10 --ensemble-size 1
    ;;
5)  echo "LegNet Yeast ground truth (resume)"
    uv run --no-sync python experiments/exp1_1_scaling.py \
        --task yeast --student legnet \
        --oracle ground_truth --reservoir genomic \
        --n-replicates 3 --seed 42 \
        --output-dir "outputs/exp0_oracle_scaling_v4/yeast/legnet_ground_truth" \
        --training-sizes ${YEAST_SIZES} --epochs 80 \
        --early-stop-patience 10 --ensemble-size 1
    ;;
*)  echo "ERROR: unknown task ${T}"; exit 1 ;;
esac

echo "=== task=${T} DONE — $(date) ==="
