#!/bin/bash
# Experiment 1.1: New reservoir strategies.
#
# Tests curriculum, uncertainty-guided, and mixed-pool strategies.
# (Adaptive selection requires special handling — not included here.)
#
# Array tasks:
#   0: curriculum_easy_first
#   1: curriculum_random (control)
#   2: uncertainty_guided
#   3: uncertainty_balanced
#   4: mixed_motif_snv
#   5: mixed_motif_prm
#
# Usage:
#   TASK=k562 STUDENT=dream_rnn ORACLE=dream_rnn sbatch scripts/slurm/exp1_1_new_strategies.sh
#
#SBATCH --job-name=exp1_1_new
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

TASK="${TASK:-k562}"
STUDENT="${STUDENT:-dream_rnn}"
ORACLE="${ORACLE:-default}"
N_REPLICATES="${N_REPLICATES:-3}"
SEED="${SEED:-42}"

STRATEGIES=(
    "curriculum_easy_first"
    "curriculum_random"
    "uncertainty_guided"
    "uncertainty_balanced"
    "mixed_motif_snv"
    "mixed_motif_prm"
)

RESERVOIR="${STRATEGIES[$SLURM_ARRAY_TASK_ID]}"

echo "=== Experiment 1.1: New Strategies ==="
echo "Task: ${TASK}, Student: ${STUDENT}, Oracle: ${ORACLE}"
echo "Reservoir: ${RESERVOIR}"
echo "Node: $SLURMD_NODENAME  Date: $(date)"

if [[ "${ORACLE}" == "default" ]]; then
    OUT_DIR="outputs/exp1_1/${TASK}/${STUDENT}"
else
    OUT_DIR="outputs/exp1_1/${TASK}/${STUDENT}_${ORACLE}"
fi

# Run small tier (1k-50k)
echo "--- Small tier (1k-50k) ---"
uv run --no-sync python experiments/exp1_1_scaling.py \
    --task "${TASK}" \
    --student "${STUDENT}" \
    --oracle "${ORACLE}" \
    --reservoir "${RESERVOIR}" \
    --n-replicates "${N_REPLICATES}" \
    --seed "${SEED}" \
    --output-dir "${OUT_DIR}" \
    --training-sizes 1000 5000 10000 20000 50000 \
    --epochs 80 \
    --ensemble-size 5 \
    --early-stop-patience 10

# Run large tier (100k-500k)
echo "--- Large tier (100k-500k) ---"
uv run --no-sync python experiments/exp1_1_scaling.py \
    --task "${TASK}" \
    --student "${STUDENT}" \
    --oracle "${ORACLE}" \
    --reservoir "${RESERVOIR}" \
    --n-replicates "${N_REPLICATES}" \
    --seed "${SEED}" \
    --output-dir "${OUT_DIR}" \
    --training-sizes 100000 200000 500000 \
    --epochs 50 \
    --ensemble-size 3 \
    --early-stop-patience 10

echo "Done: $(date)"
