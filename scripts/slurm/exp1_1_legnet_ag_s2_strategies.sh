#!/bin/bash
# Exp 1.1: LegNet student + AG S2 LIVE oracle — all strategies.
# Uses live AG S2 inference (not lookup) so novel sequences get real labels.
#
# Priority strategies for Peter's talk:
#   Baselines: random, genomic, dinuc_shuffle, gc_matched
#   Mutagenesis: prm_5pct, prm_10pct
#   Augmentation: evoaug_structural, evoaug_heavy
#   Recombination: recombination_uniform
#   Motif: motif_planted, motif_grammar
#
# Array (4 tasks, ~3 strategies each):
#   0: random, genomic, dinuc_shuffle
#   1: gc_matched, prm_5pct, prm_10pct
#   2: evoaug_structural, evoaug_heavy, recombination_uniform
#   3: motif_planted, motif_grammar, snv
#
# REQUIRES H100 for AG S2 oracle (JAX encoder inference).
#
# Submit:
#   /cm/shared/apps/slurm/current/bin/sbatch --array=0-3 scripts/slurm/exp1_1_legnet_ag_s2_strategies.sh
#
#SBATCH --job-name=e1_lgnt_s2
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=200G

set -euo pipefail

set +u; source /etc/profile.d/modules.sh; set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
source scripts/slurm/setup_hpc_deps.sh
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_command_buffer="

T=$SLURM_ARRAY_TASK_ID
echo "=== Exp1.1 LegNet + AG S2 live oracle task=${T} node=${SLURMD_NODENAME} $(date) ==="

GROUPS=(
    "random genomic dinuc_shuffle"
    "gc_matched prm_5pct prm_10pct"
    "evoaug_structural evoaug_heavy recombination_uniform"
    "motif_planted motif_grammar snv"
)
STRATEGIES=${GROUPS[$T]}

echo "Strategies: ${STRATEGIES}"

# Small tier (1k-50k): HP sweep with ensemble=5
echo "--- Small tier ---"
uv run --no-sync python experiments/exp1_1_scaling.py \
    --task k562 --student legnet --oracle ag_s2 \
    --reservoir ${STRATEGIES} \
    --n-replicates 3 --seed 42 \
    --output-dir "outputs/exp1_1/k562/legnet_ag_s2" \
    --training-sizes 1000 5000 10000 20000 50000 \
    --lr 0.001 --batch-size 1024 \
    --epochs 80 --ensemble-size 1 --early-stop-patience 10 \
    --save-predictions || true

# Large tier (100k-296k)
echo "--- Large tier ---"
uv run --no-sync python experiments/exp1_1_scaling.py \
    --task k562 --student legnet --oracle ag_s2 \
    --reservoir ${STRATEGIES} \
    --n-replicates 3 --seed 42 \
    --output-dir "outputs/exp1_1/k562/legnet_ag_s2" \
    --training-sizes 100000 200000 \
    --lr 0.001 --batch-size 1024 \
    --epochs 50 --ensemble-size 1 --early-stop-patience 10 \
    --save-predictions || true

echo "=== task=${T} DONE — $(date) ==="
