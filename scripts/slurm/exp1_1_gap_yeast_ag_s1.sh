#!/bin/bash
# Experiment 1.1 gap-fill: Yeast AG S1 — 6 missing strategies.
#
# Missing strategies (0 results):
#   prm_5pct, prm_10pct, prm_uniform_1_10, dinuc_shuffle, gc_matched, motif_planted
#
# exp1_1_scaling.py has built-in skip logic for existing results.
# AG S1 uses JAX -> needs H100 for best performance.
#
# Array: 2 tasks running 3 strategies each.
#
# Submit:
#   /cm/shared/apps/slurm/current/bin/sbatch scripts/slurm/exp1_1_gap_yeast_ag_s1.sh
#
#SBATCH --job-name=exp1_1_gap_ag_s1_yeast
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=200G
#SBATCH --array=0-1

set -euo pipefail

set +u
source /etc/profile.d/modules.sh
set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
source scripts/slurm/setup_hpc_deps.sh

export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_command_buffer="

TASK="yeast"
STUDENT="alphagenome_yeast_s1"
ORACLE="ag"
N_REPLICATES=3
SEED=42
OUT_DIR="outputs/exp1_1/${TASK}/${STUDENT}_${ORACLE}"

# 6 missing strategies in 2 groups of 3
ALL_RESERVOIRS=(
    "prm_5pct prm_10pct prm_uniform_1_10"
    "dinuc_shuffle gc_matched motif_planted"
)
RESERVOIR_GROUP=${ALL_RESERVOIRS[$SLURM_ARRAY_TASK_ID]}

echo "=== Exp1.1 Gap-Fill: Yeast AG S1 ==="
echo "Reservoirs: ${RESERVOIR_GROUP}"
echo "Node: ${SLURMD_NODENAME}  Date: $(date)"
echo "Array task: ${SLURM_ARRAY_TASK_ID}/1"

# Small tier (1k-50k)
echo "--- Small tier (1k-50k) ---"
uv run --no-sync python experiments/exp1_1_scaling.py \
    --task "${TASK}" \
    --student "${STUDENT}" \
    --oracle "${ORACLE}" \
    --reservoir ${RESERVOIR_GROUP} \
    --n-replicates "${N_REPLICATES}" \
    --seed "${SEED}" \
    --output-dir "${OUT_DIR}" \
    --training-sizes 1000 5000 10000 20000 50000 \
    --epochs 80 \
    --ensemble-size 5 \
    --early-stop-patience 10

# Large tier (100k-500k)
echo "--- Large tier (100k-500k) ---"
uv run --no-sync python experiments/exp1_1_scaling.py \
    --task "${TASK}" \
    --student "${STUDENT}" \
    --oracle "${ORACLE}" \
    --reservoir ${RESERVOIR_GROUP} \
    --n-replicates "${N_REPLICATES}" \
    --seed "${SEED}" \
    --output-dir "${OUT_DIR}" \
    --training-sizes 100000 200000 500000 \
    --epochs 50 \
    --ensemble-size 3 \
    --early-stop-patience 10 \
    --transfer-hp-from 50000

echo "Done: $(date)"
