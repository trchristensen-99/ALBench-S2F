#!/bin/bash
# Experiment 1.2: Acquisition function benchmarking.
# Array job: packs acquisition functions into groups for efficient slot usage.
#
# Each array task runs one reservoir x all acquisitions for one regime.
# With 7 acquisition functions x 3 regimes = 21 combos; we pack 7 acqs
# per task and use 3 array tasks (one per regime).
#
# Usage:
#   TASK=k562 STUDENT=dream_rnn RESERVOIR=random sbatch scripts/slurm/exp1_2_acquisition.sh
#   TASK=k562 STUDENT=dream_rnn RESERVOIR=random REGIME=small sbatch scripts/slurm/exp1_2_acquisition.sh
#
#SBATCH --job-name=exp1_2_acq
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=200G
#SBATCH --array=0-2

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
RESERVOIR="${RESERVOIR:-random}"
N_REPLICATES="${N_REPLICATES:-3}"
SEED="${SEED:-42}"
POOL_RATIO="${POOL_RATIO:-10}"
ENSEMBLE_SIZE="${ENSEMBLE_SIZE:-5}"
EPOCHS="${EPOCHS:-80}"
EARLY_STOP="${EARLY_STOP:-}"

# Map array task ID to regime
ALL_REGIMES=(small medium large)
REGIME="${REGIME:-${ALL_REGIMES[$SLURM_ARRAY_TASK_ID]}}"

# All acquisition functions
ACQUISITIONS="random uncertainty diversity badge batchbald combined ensemble prior_knowledge"

echo "=== Experiment 1.2: Acquisition Function Benchmarking ==="
echo "Task: ${TASK}, Student: ${STUDENT}, Oracle: ${ORACLE}"
echo "Reservoir: ${RESERVOIR}, Regime: ${REGIME}"
echo "Acquisitions: ${ACQUISITIONS}"
echo "Node: $SLURMD_NODENAME  Date: $(date)"
echo "Array task: ${SLURM_ARRAY_TASK_ID}/2"

# Build output dir
if [[ "${ORACLE}" == "default" ]]; then
    OUT_DIR="outputs/exp1_2/${TASK}/${STUDENT}"
else
    OUT_DIR="outputs/exp1_2/${TASK}/${STUDENT}_${ORACLE}"
fi

EXTRA_ARGS=""
if [[ -n "${EARLY_STOP}" ]]; then
    EXTRA_ARGS="${EXTRA_ARGS} --early-stop-patience ${EARLY_STOP}"
fi

uv run --no-sync python experiments/exp1_2_acquisition.py \
    --task "${TASK}" \
    --student "${STUDENT}" \
    --oracle "${ORACLE}" \
    --reservoir "${RESERVOIR}" \
    --acquisition ${ACQUISITIONS} \
    --regime "${REGIME}" \
    --pool-ratio "${POOL_RATIO}" \
    --n-replicates "${N_REPLICATES}" \
    --seed "${SEED}" \
    --output-dir "${OUT_DIR}" \
    --ensemble-size "${ENSEMBLE_SIZE}" \
    --epochs "${EPOCHS}" \
    ${EXTRA_ARGS}

echo "Done: $(date)"
