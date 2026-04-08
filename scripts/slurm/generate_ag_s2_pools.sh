#!/bin/bash
# Generate pre-cached labeled pools for all priority strategies with AG S2 oracle.
# Each pool: 296K-500K sequences labeled by the live AG S2 10-fold ensemble.
# After this, LegNet training can run on V100 without JAX.
#
# Array tasks (18 strategies grouped by tier):
#   Tier 1 (must have):
#     0: random           1: genomic          2: dinuc_shuffle
#     3: prm_5pct         4: prm_10pct        5: evoaug_structural
#   Tier 2 (important):
#     6: gc_matched        7: recombination_uniform  8: motif_planted
#     9: motif_grammar    10: motif_clustering_mutant 11: evoaug_heavy
#   Tier 3 (nice to have):
#    12: activity_stratified_oracle  13: prm_1pct  14: prm_20pct
#    15: prm_50pct  16: snv  17: prm_uniform_1_10
#
# Submit all:
#   /cm/shared/apps/slurm/current/bin/sbatch --array=0-17 scripts/slurm/generate_ag_s2_pools.sh
# Submit tier 1 only:
#   /cm/shared/apps/slurm/current/bin/sbatch --array=0-5 scripts/slurm/generate_ag_s2_pools.sh
#
#SBATCH --job-name=gen_pools
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

STRATEGIES=(
    random genomic dinuc_shuffle prm_5pct prm_10pct evoaug_structural
    gc_matched recombination_uniform motif_planted motif_grammar motif_clustering_mutant evoaug_heavy
    activity_stratified_oracle prm_1pct prm_20pct prm_50pct snv prm_uniform_1_10
)

STRATEGY="${STRATEGIES[$T]}"
echo "=== gen_pools task=${T} strategy=${STRATEGY} node=${SLURMD_NODENAME} $(date) ==="

uv run --no-sync python scripts/generate_labeled_pools.py \
    --task k562 \
    --oracle ag_s2 \
    --reservoir "${STRATEGY}" \
    --pool-size 500000 \
    --seed 42

echo "=== Done: $(date) ==="
