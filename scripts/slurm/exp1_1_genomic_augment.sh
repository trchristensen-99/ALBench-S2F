#!/bin/bash
# Experiment 1.1: Genomic base + augmentation paradigm.
#
# Tests whether augmenting a genomic base with synthetic sequences from
# different strategies is better than pure genomic at the same total size.
#
# Design:
#   - Mixed reservoir: 50% genomic + 50% strategy X
#   - Training sizes: 20K, 50K, 100K, 200K, 500K total
#     (i.e., 10K+10K, 25K+25K, 50K+50K, 100K+100K, 250K+250K)
#   - Baseline comparison: pure genomic at same total sizes
#     (already exists from prior exp1_1 runs)
#   - Student: LegNet (priority)
#   - Oracle: default (AlphaGenome)
#
# Augmentation strategies tested:
#   0: mixed_genomic_random
#   1: mixed_genomic_dinuc_shuffle
#   2: mixed_genomic_evoaug
#   3: mixed_genomic_prm
#   4: mixed_genomic_recombination
#   5: mixed_genomic_snv
#
# Submit:
#   /cm/shared/apps/slurm/current/bin/sbatch --array=0-5 scripts/slurm/exp1_1_genomic_augment.sh
#
# To run a single strategy (e.g., random only):
#   /cm/shared/apps/slurm/current/bin/sbatch --array=0 scripts/slurm/exp1_1_genomic_augment.sh
#
#SBATCH --job-name=exp1_1_geno_aug
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
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

TASK="k562"
STUDENT="legnet"
ORACLE="default"
N_REPLICATES=3
SEED=42
OUT_DIR="outputs/exp1_1/${TASK}/${STUDENT}"

# Each array task runs one mixed genomic+strategy reservoir
RESERVOIRS=(
    "mixed_genomic_random"
    "mixed_genomic_dinuc_shuffle"
    "mixed_genomic_evoaug"
    "mixed_genomic_prm"
    "mixed_genomic_recombination"
    "mixed_genomic_snv"
)
RESERVOIR=${RESERVOIRS[$SLURM_ARRAY_TASK_ID]}

echo "=== Exp1.1 Genomic Base + Augment ==="
echo "Reservoir: ${RESERVOIR}"
echo "Student: ${STUDENT}, Oracle: ${ORACLE}"
echo "Node: ${SLURMD_NODENAME}  Date: $(date)"
echo "Array task: ${SLURM_ARRAY_TASK_ID}/5"
echo "Output: ${OUT_DIR}"

# Small tier (20k-50k): full HP sweep
echo "--- Small tier (20k-50k) ---"
uv run --no-sync python experiments/exp1_1_scaling.py \
    --task "${TASK}" \
    --student "${STUDENT}" \
    --oracle "${ORACLE}" \
    --reservoir ${RESERVOIR} \
    --n-replicates "${N_REPLICATES}" \
    --seed "${SEED}" \
    --output-dir "${OUT_DIR}" \
    --training-sizes 20000 50000 \
    --epochs 80 \
    --ensemble-size 5 \
    --early-stop-patience 10

# Large tier (100k-500k): transfer HP from n=50k
echo "--- Large tier (100k-500k) ---"
uv run --no-sync python experiments/exp1_1_scaling.py \
    --task "${TASK}" \
    --student "${STUDENT}" \
    --oracle "${ORACLE}" \
    --reservoir ${RESERVOIR} \
    --n-replicates "${N_REPLICATES}" \
    --seed "${SEED}" \
    --output-dir "${OUT_DIR}" \
    --training-sizes 100000 200000 500000 \
    --epochs 50 \
    --ensemble-size 3 \
    --early-stop-patience 10 \
    --transfer-hp-from 50000

echo "=== ${RESERVOIR} DONE — $(date) ==="
