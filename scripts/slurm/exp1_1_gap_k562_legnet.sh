#!/bin/bash
# Experiment 1.1 gap-fill: K562 LegNet missing strategies.
#
# Currently only 5 strategy dirs exist (dinuc_shuffle, evoaug_heavy, gc_matched,
# genomic, random) — most with incomplete results.
# Missing 16 strategies completely.
#
# exp1_1_scaling.py has built-in skip logic for existing results.
# PyTorch model -> can use V100 nodes.
#
# Array: 7 tasks, each running 3 strategies (same grouping as exp1_1_scaling.sh).
#
# Submit:
#   /cm/shared/apps/slurm/current/bin/sbatch scripts/slurm/exp1_1_gap_k562_legnet.sh
#
#SBATCH --job-name=exp1_1_gap_legnet_k562
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
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

TASK="k562"
STUDENT="legnet"
ORACLE="default"
N_REPLICATES=3
SEED=42
OUT_DIR="outputs/exp1_1/${TASK}/${STUDENT}"

# 21 reservoirs in 7 groups of 3 (same grouping as main exp1_1_scaling.sh)
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

echo "=== Exp1.1 Gap-Fill: K562 LegNet ==="
echo "Reservoirs: ${RESERVOIR_GROUP}"
echo "Node: ${SLURMD_NODENAME}  Date: $(date)"
echo "Array task: ${SLURM_ARRAY_TASK_ID}/6"

# Small tier (1k-50k): full HP sweep
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

# Large tier (100k-500k): transfer HP from n=50k
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
