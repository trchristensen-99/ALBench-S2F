#!/bin/bash
# Train ISE fitness predictor models for Experiment 1.1.
# Trains all 4 variants (dream_rnn x {10%,100%} + ag_s1 x {10%,100%}) for one task.
#
# Usage:
#   TASK=k562 sbatch scripts/slurm/train_ise_fitness_models.sh
#   TASK=yeast sbatch scripts/slurm/train_ise_fitness_models.sh
#
#SBATCH --job-name=train_ise_fitness
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --time=06:00:00
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

TASK="${TASK:-k562}"
SEED="${SEED:-42}"

echo "=== Training ISE Fitness Models ==="
echo "Task: ${TASK}, Seed: ${SEED}"
echo "Node: $SLURMD_NODENAME  Date: $(date)"

# Train all 4 variants sequentially.
# Oracle is loaded fresh for each run (freed after labeling).

echo ""
echo "--- DREAM-RNN 10% ---"
uv run --no-sync python experiments/train_ise_fitness_models.py \
    --task "${TASK}" \
    --model-type dream_rnn \
    --train-fraction 0.1 \
    --seed "${SEED}"

echo ""
echo "--- DREAM-RNN 100% ---"
uv run --no-sync python experiments/train_ise_fitness_models.py \
    --task "${TASK}" \
    --model-type dream_rnn \
    --train-fraction 1.0 \
    --seed "${SEED}"

echo ""
echo "--- AG-S1 10% ---"
uv run --no-sync python experiments/train_ise_fitness_models.py \
    --task "${TASK}" \
    --model-type ag_s1 \
    --train-fraction 0.1 \
    --seed "${SEED}"

echo ""
echo "--- AG-S1 100% ---"
uv run --no-sync python experiments/train_ise_fitness_models.py \
    --task "${TASK}" \
    --model-type ag_s1 \
    --train-fraction 1.0 \
    --seed "${SEED}"

echo ""
echo "=== All ISE fitness models trained for ${TASK} ==="
echo "Outputs: outputs/ise_fitness_models/${TASK}/"
echo "Done: $(date)"
