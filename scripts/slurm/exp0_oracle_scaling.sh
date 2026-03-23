#!/bin/bash
# Experiment 0: Oracle-label data scaling curves.
#
# Trains student models on oracle pseudo-labels at various fractions
# of the full training pool. Uses "random" reservoir (genomic pool subsample).
#
# Fractions match real-label experiments for consistency:
#   K562 (pool=319,742):  1%, 2%, 5%, 10%, 20%, 50%, 100%
#   Yeast (pool=6,065,324): 0.1%, 0.2%, 0.5%, 1%, 2%, 5%, 10%, 20%, 50%, 100%
#
# Usage:
#   TASK=k562 STUDENT=dream_cnn   sbatch scripts/slurm/exp0_oracle_scaling.sh
#   TASK=k562 STUDENT=dream_rnn   sbatch scripts/slurm/exp0_oracle_scaling.sh
#   TASK=k562 STUDENT=alphagenome_k562_s1  sbatch scripts/slurm/exp0_oracle_scaling.sh
#   TASK=yeast STUDENT=dream_cnn  sbatch scripts/slurm/exp0_oracle_scaling.sh
#   TASK=yeast STUDENT=dream_rnn  sbatch scripts/slurm/exp0_oracle_scaling.sh
#   TASK=yeast STUDENT=alphagenome_yeast_s1  sbatch scripts/slurm/exp0_oracle_scaling.sh
#
#SBATCH --job-name=exp0_oracle
#SBATCH --output=logs/%x-%A.out
#SBATCH --error=logs/%x-%A.err
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

TASK="${TASK:-k562}"
STUDENT="${STUDENT:-dream_cnn}"
ORACLE="${ORACLE:-default}"
N_REPLICATES="${N_REPLICATES:-3}"
SEED="${SEED:-42}"

# Task-specific training sizes matching real-label experiment fractions
if [[ "${TASK}" == "k562" ]]; then
    # K562 pool = 319,742. Fractions: 1% 2% 5% 10% 20% 50% 100%
    TRAINING_SIZES="3197 6395 15987 31974 63949 159871 319742"
    TRANSFER_HP_FROM=31974  # Transfer HP from 10% fraction
elif [[ "${TASK}" == "yeast" ]]; then
    # Yeast pool = 6,065,324. Fractions: 0.1% 0.2% 0.5% 1% 2% 5% 10% 20% 50% 100%
    TRAINING_SIZES="6065 12131 30327 60653 121307 303266 606532 1213065 3032662 6065324"
    TRANSFER_HP_FROM=60653  # Transfer HP from 1% fraction
else
    echo "ERROR: Unknown task ${TASK}"
    exit 1
fi

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
    --transfer-hp-from "${TRANSFER_HP_FROM}"

echo "Done: $(date)"
