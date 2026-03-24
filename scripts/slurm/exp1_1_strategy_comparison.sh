#!/bin/bash
# Experiment 1.1: Compare biologically sensible data generation strategies.
#
# Tests the PI's key question: what is the most informative source of data?
# Uses DREAM-CNN (fast, good performance) at a moderate training size.
#
# Strategies (array tasks):
#   0: random              - Pure random sequences (baseline)
#   1: dinuc_shuffle        - Dinucleotide-preserving shuffle of genomic seqs
#   2: genomic              - Uniform random from genomic pool
#   3: motif_clustering     - Cluster by regulatory architecture, sample uniformly
#   4: motif_clustering_mutant - Cluster + light mutagenesis (3% rate)
#   5: prm_5pct             - 5% point mutagenesis of genomic seqs
#   6: prm_10pct            - 10% point mutagenesis of genomic seqs
#   7: activity_stratified_oracle - Stratify by expression level (oracle labels)
#   8: gc_matched           - GC-content matched to genomic pool
#   9: evoaug_structural    - Structural mutations (deletions, inversions, etc.)
#
# Each runs 3 replicates × 2 HP configs (bs=512,1024) at N=31,974 (10% of K562).
# Quick: DREAM-CNN at this size takes ~5min per replicate.
#
#SBATCH --job-name=exp1_1_strat
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=default
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=200G
#SBATCH --array=0-9

set -euo pipefail

set +u
source /etc/profile.d/modules.sh
set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
source scripts/slurm/setup_hpc_deps.sh

export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_command_buffer="

STRATEGIES=(
    "random"
    "dinuc_shuffle"
    "genomic"
    "motif_clustering"
    "motif_clustering_mutant"
    "prm_5pct"
    "prm_10pct"
    "activity_stratified_oracle"
    "gc_matched"
    "evoaug_structural"
)

STRATEGY="${STRATEGIES[$SLURM_ARRAY_TASK_ID]}"
STUDENT="${STUDENT:-dream_cnn}"
TASK="${TASK:-k562}"
N_TRAIN="${N_TRAIN:-31974}"

echo "=== Experiment 1.1: Strategy Comparison ==="
echo "Strategy: ${STRATEGY}, Student: ${STUDENT}, Task: ${TASK}, N=${N_TRAIN}"
echo "Node: $SLURMD_NODENAME  Date: $(date)"

uv run --no-sync python experiments/exp1_1_scaling.py \
    --task "${TASK}" \
    --student "${STUDENT}" \
    --oracle default \
    --reservoir "${STRATEGY}" \
    --n-replicates 3 \
    --seed 42 \
    --output-dir "outputs/exp1_1_strategy_comparison/${TASK}/${STUDENT}" \
    --training-sizes "${N_TRAIN}" \
    --epochs 80 \
    --ensemble-size 3 \
    --early-stop-patience 10

echo "Done: $(date)"
