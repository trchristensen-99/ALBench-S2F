#!/bin/bash
# Experiment 1.1: Reservoir sampling scaling laws.
# Array job: packs 3 reservoirs per task for efficient slot usage.
#
# Splits training sizes into tiers to fit within SLURM time limits:
#   TIER=small (default): n <= 50,000  — full HP sweep, ~10h/reservoir
#   TIER=large:           n = 100k–500k — reduced HP + early stopping, ~12h/reservoir
#
# With 3 reservoirs per task, each task runs ~30h (small) or ~36h (large).
# 7 array tasks per job × 4 jobs = 28 tasks → fits within 20 concurrent slots
# with room to interleave jobs from different tiers.
#
# Usage:
#   TASK=k562 STUDENT=dream_rnn ORACLE=dream_rnn sbatch scripts/slurm/exp1_1_scaling.sh
#   TASK=k562 STUDENT=dream_rnn ORACLE=dream_rnn TIER=large sbatch scripts/slurm/exp1_1_scaling.sh
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
#SBATCH --array=0-6

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
TIER="${TIER:-small}"

# 21 reservoirs packed into 7 groups of 3
ALL_RESERVOIRS=(
    "random genomic prm_1pct"
    "prm_5pct prm_10pct prm_uniform_1_10"
    "dinuc_shuffle gc_matched motif_planted"
    "recombination_uniform recombination_2pt prm_20pct"
    "prm_50pct motif_grammar motif_grammar_tight"
    "evoaug_structural evoaug_heavy ise_maximize"
    "ise_diverse_targets ise_target_high snv"
)
RESERVOIR_GROUP=${ALL_RESERVOIRS[$SLURM_ARRAY_TASK_ID]}

echo "=== Experiment 1.1: Scaling Laws ==="
echo "Task: ${TASK}, Student: ${STUDENT}, Oracle: ${ORACLE}"
echo "Tier: ${TIER}, Reservoirs: ${RESERVOIR_GROUP}"
echo "Node: $SLURMD_NODENAME  Date: $(date)"
echo "Array task: ${SLURM_ARRAY_TASK_ID}/6"

# Build output dir: includes oracle suffix when non-default
if [[ "${ORACLE}" == "default" ]]; then
    OUT_DIR="outputs/exp1_1/${TASK}/${STUDENT}"
else
    OUT_DIR="outputs/exp1_1/${TASK}/${STUDENT}_${ORACLE}"
fi

# Tier-specific settings
if [[ "${TIER}" == "large" ]]; then
    # Large sizes: transfer HP from n=50k + early stopping + fewer ensemble members
    TRAINING_SIZES="100000 200000 500000"
    EPOCHS=50
    EARLY_STOP=10
    ENSEMBLE_SIZE=3
    TRANSFER_HP=50000
else
    # Small sizes: full sweep, early stopping saves time at larger N
    TRAINING_SIZES="1000 5000 10000 20000 50000"
    EPOCHS=80
    EARLY_STOP=10
    ENSEMBLE_SIZE=5
fi

EXTRA_ARGS=""
if [[ -n "${EARLY_STOP}" ]]; then
    EXTRA_ARGS="${EXTRA_ARGS} --early-stop-patience ${EARLY_STOP}"
fi
if [[ -n "${TRANSFER_HP:-}" ]]; then
    EXTRA_ARGS="${EXTRA_ARGS} --transfer-hp-from ${TRANSFER_HP}"
fi

# Run all reservoirs in this group sequentially (they share the oracle + pool)
uv run --no-sync python experiments/exp1_1_scaling.py \
    --task "${TASK}" \
    --student "${STUDENT}" \
    --oracle "${ORACLE}" \
    --reservoir ${RESERVOIR_GROUP} \
    --n-replicates "${N_REPLICATES}" \
    --seed "${SEED}" \
    --output-dir "${OUT_DIR}" \
    --training-sizes ${TRAINING_SIZES} \
    --epochs "${EPOCHS}" \
    --ensemble-size "${ENSEMBLE_SIZE}" \
    ${EXTRA_ARGS}

echo "Done: $(date)"
