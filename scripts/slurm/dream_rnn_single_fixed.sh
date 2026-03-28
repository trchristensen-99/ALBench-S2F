#!/bin/bash
# DREAM-RNN single-model re-run for K562 with lower LR (0.001 vs 0.005).
# The original single-model runs (dream_rnn_single_model.sh) used lr=0.005
# which caused 2/3 K562 seeds to diverge. The 3-ensemble version is stable at
# lr=0.005 because the ensemble averaging dampens gradient noise; single models
# need a lower LR.
#
# Array mapping: 0-2 = K562 seeds 0, 1, 2
#
# Usage:
#   sbatch --array=0-2 scripts/slurm/dream_rnn_single_fixed.sh
#
#SBATCH --job-name=drnn_single_v2
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=default
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=200G

set -euo pipefail

set +u
source /etc/profile.d/modules.sh
set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
source scripts/slurm/setup_hpc_deps.sh

export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_command_buffer="

SEEDS=(0 1 2)
SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"

echo "=== DREAM-RNN single-model v2 (ensemble_size=1, lr=0.001) ==="
echo "Seed: ${SEED}"
echo "Array task: ${SLURM_ARRAY_TASK_ID}"
echo "Node: $SLURMD_NODENAME  Date: $(date)"

uv run --no-sync python experiments/exp1_1_scaling.py \
    --task k562 \
    --student dream_rnn \
    --oracle ground_truth \
    --reservoir genomic \
    --n-replicates 1 \
    --no-hp-sweep \
    --lr 0.001 \
    --seed "${SEED}" \
    --output-dir "outputs/dream_rnn_k562_single_v2/seed_${SEED}" \
    --training-sizes 319742 \
    --epochs 80 \
    --ensemble-size 1 \
    --early-stop-patience 10

echo "Done: $(date)"
