#!/bin/bash
# Experiment 0: Oracle-label data scaling curves.
#
# Trains student models on oracle pseudo-labels at various fractions
# of the full training pool. Uses "random" reservoir (genomic pool subsample).
#
# Usage:
#   TASK=k562 STUDENT=dream_cnn ORACLE=default sbatch scripts/slurm/exp0_oracle_scaling.sh
#   TASK=yeast STUDENT=dream_rnn ORACLE=default sbatch scripts/slurm/exp0_oracle_scaling.sh
#   TASK=yeast STUDENT=dream_cnn ORACLE=default sbatch scripts/slurm/exp0_oracle_scaling.sh
#   TASK=k562 STUDENT=ag ORACLE=default sbatch scripts/slurm/exp0_oracle_scaling.sh
#
#SBATCH --job-name=exp0_oracle
#SBATCH --output=logs/%x-%A.out
#SBATCH --error=logs/%x-%A.err
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --time=48:00:00
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
STUDENT="${STUDENT:-dream_cnn}"
ORACLE="${ORACLE:-default}"
N_REPLICATES="${N_REPLICATES:-3}"
SEED="${SEED:-42}"

# Oracle-label scaling: use "random" reservoir at many sizes
# These correspond to fractions of the ~320k pool:
#   1k, 3k, 6k, 16k, 32k, 64k, 160k, 320k
TRAINING_SIZES="1000 3000 6000 16000 32000 64000 160000 320000"

echo "=== Experiment 0: Oracle-Label Scaling ==="
echo "Task: ${TASK}, Student: ${STUDENT}, Oracle: ${ORACLE}"
echo "Training sizes: ${TRAINING_SIZES}"
echo "Node: $SLURMD_NODENAME  Date: $(date)"

OUT_DIR="outputs/exp0_oracle_scaling_v4/${TASK}/${STUDENT}"

# Run with HP sweep at small N, transfer HP at large N
uv run --no-sync python experiments/exp1_1_scaling.py \
    --task "${TASK}" \
    --student "${STUDENT}" \
    --oracle "${ORACLE}" \
    --reservoir random \
    --n-replicates "${N_REPLICATES}" \
    --seed "${SEED}" \
    --output-dir "${OUT_DIR}" \
    --training-sizes ${TRAINING_SIZES} \
    --epochs 80 \
    --ensemble-size 3 \
    --early-stop-patience 10 \
    --transfer-hp-from 32000

echo "Done: $(date)"
