#!/bin/bash
# Experiment 1.1: Reservoir sampling scaling laws.
# Array job: one task per reservoir strategy.
# Trains DREAM-RNN students at 8 training sizes with HP sweep.
#
# Usage:
#   TASK=k562 STUDENT=dream_rnn sbatch scripts/slurm/exp1_1_scaling.sh
#   TASK=yeast STUDENT=dream_rnn sbatch scripts/slurm/exp1_1_scaling.sh
#
#SBATCH --job-name=exp1_1_scaling
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=200G
#SBATCH --array=0-5

set -euo pipefail

set +u
source /etc/profile.d/modules.sh
set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
source scripts/slurm/setup_hpc_deps.sh

export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_command_buffer="

# Configuration (override via environment)
TASK="${TASK:-k562}"
STUDENT="${STUDENT:-dream_rnn}"
ORACLE="${ORACLE:-default}"
N_REPLICATES="${N_REPLICATES:-3}"
SEED="${SEED:-42}"

RESERVOIRS=(random genomic prm_1pct prm_5pct prm_10pct prm_uniform_1_10)
RESERVOIR=${RESERVOIRS[$SLURM_ARRAY_TASK_ID]}

echo "=== Experiment 1.1: Scaling Laws ==="
echo "Task: ${TASK}, Student: ${STUDENT}, Oracle: ${ORACLE}, Reservoir: ${RESERVOIR}"
echo "Node: $SLURMD_NODENAME  Date: $(date)"
echo "Array task: ${SLURM_ARRAY_TASK_ID}/5"

# Build output dir: includes oracle suffix when non-default
if [[ "${ORACLE}" == "default" ]]; then
    OUT_DIR="outputs/exp1_1/${TASK}/${STUDENT}"
else
    OUT_DIR="outputs/exp1_1/${TASK}/${STUDENT}_${ORACLE}"
fi

uv run --no-sync python experiments/exp1_1_scaling.py \
    --task "${TASK}" \
    --student "${STUDENT}" \
    --oracle "${ORACLE}" \
    --reservoir "${RESERVOIR}" \
    --n-replicates "${N_REPLICATES}" \
    --seed "${SEED}" \
    --output-dir "${OUT_DIR}"

echo "Done: $(date)"
