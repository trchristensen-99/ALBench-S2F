#!/bin/bash
# Experiment 1.1: Two-tier ISE reservoir experiments.
#
# Tier 1 (Oracle): ISE uses oracle as fitness function (upper bound).
#   Already handled by base ISE names (ise_maximize, ise_diverse_targets, ise_target_high).
#
# Tier 2 (Realistic): ISE uses a model trained on 10% of data as fitness.
#   Uses suffix variants: ise_maximize_dream10, ise_diverse_targets_dream10, etc.
#
# This script runs BOTH tiers for a given oracle-student config.
# Array tasks:
#   0-2: Oracle-guided ISE (3 strategies)
#   3-5: DREAM-10% fitness ISE (3 strategies)
#   6-8: AG-10% fitness ISE (3 strategies, K562 only)
#
# Usage:
#   TASK=k562 STUDENT=dream_rnn ORACLE=dream_rnn sbatch scripts/slurm/exp1_1_ise_twotier.sh
#   TASK=yeast STUDENT=dream_rnn ORACLE=dream_rnn sbatch scripts/slurm/exp1_1_ise_twotier.sh
#
#SBATCH --job-name=exp1_1_ise
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=200G
#SBATCH --array=0-8

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

# ISE strategy names by array task
ISE_STRATEGIES=(
    "ise_maximize"            # 0: oracle
    "ise_diverse_targets"     # 1: oracle
    "ise_target_high"         # 2: oracle
    "ise_maximize_dream10"    # 3: DREAM 10% fitness
    "ise_diverse_targets_dream10"  # 4: DREAM 10% fitness
    "ise_target_high_dream10"      # 5: DREAM 10% fitness
    "ise_maximize_ag10"       # 6: AG 10% fitness (K562 only)
    "ise_diverse_targets_ag10"     # 7: AG 10% fitness (K562 only)
    "ise_target_high_ag10"         # 8: AG 10% fitness (K562 only)
)

RESERVOIR="${ISE_STRATEGIES[$SLURM_ARRAY_TASK_ID]}"

# Skip AG fitness tasks for yeast (no AG fitness model available)
if [[ "${TASK}" == "yeast" ]] && [[ "$SLURM_ARRAY_TASK_ID" -ge 6 ]]; then
    echo "Skipping ${RESERVOIR} for yeast (no AG fitness model)"
    exit 0
fi

echo "=== Experiment 1.1: ISE Two-Tier ==="
echo "Task: ${TASK}, Student: ${STUDENT}, Oracle: ${ORACLE}"
echo "Reservoir: ${RESERVOIR}"
echo "Node: $SLURMD_NODENAME  Date: $(date)"
echo "Array task: ${SLURM_ARRAY_TASK_ID}/8"

if [[ "${ORACLE}" == "default" ]]; then
    OUT_DIR="outputs/exp1_1/${TASK}/${STUDENT}"
else
    OUT_DIR="outputs/exp1_1/${TASK}/${STUDENT}_${ORACLE}"
fi

# Run all training sizes for this single ISE strategy
# Small tier first (1k-50k), then large (100k-500k)
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
