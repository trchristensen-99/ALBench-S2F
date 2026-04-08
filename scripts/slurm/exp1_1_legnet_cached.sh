#!/bin/bash
# Exp 1.1: LegNet student trained on pre-cached AG S2 oracle pools.
# Runs on V100 (no JAX needed — oracle labels are pre-cached).
#
# HP sweep: 4 configs (lr=[0.001,0.005] x bs=[512,1024]) x 3 seeds at small N.
# Large N: transfer best HP from n=50K.
# Sizes: 1K, 5K, 10K, 20K, 50K, 100K, 200K, 296K
#
# Array tasks (one per strategy, matching generate_ag_s2_pools.sh):
#   0: random  1: genomic  2: dinuc_shuffle  3: prm_5pct  4: prm_10pct
#   5: evoaug_structural  6: gc_matched  7: recombination_uniform
#   8: motif_planted  9: motif_grammar  10: motif_clustering_mutant  11: evoaug_heavy
#
# Submit (after pools are generated):
#   /cm/shared/apps/slurm/current/bin/sbatch --array=0-11 scripts/slurm/exp1_1_legnet_cached.sh
#
#SBATCH --job-name=e1_lgnt_pool
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=64G

set -euo pipefail

set +u; source /etc/profile.d/modules.sh; set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
source scripts/slurm/setup_hpc_deps.sh

T=$SLURM_ARRAY_TASK_ID

STRATEGIES=(
    random genomic dinuc_shuffle prm_5pct prm_10pct evoaug_structural
    gc_matched recombination_uniform motif_planted motif_grammar motif_clustering_mutant evoaug_heavy
)

STRATEGY="${STRATEGIES[$T]}"
POOL_DIR="outputs/labeled_pools/k562/ag_s2"
OUT_DIR="outputs/exp1_1/k562/legnet_ag_s2"

echo "=== LegNet + AG S2 cached pool: ${STRATEGY} node=${SLURMD_NODENAME} $(date) ==="

# Small tier (1K-50K): full HP sweep
echo "--- Small tier (HP sweep, 4 configs x 3 seeds) ---"
uv run --no-sync python experiments/exp1_1_scaling.py \
    --task k562 --student legnet --oracle ag_s2 \
    --reservoir "${STRATEGY}" \
    --pool-base-dir "${POOL_DIR}" \
    --n-replicates 3 --seed 42 \
    --output-dir "${OUT_DIR}" \
    --training-sizes 1000 5000 10000 20000 50000 \
    --epochs 80 --ensemble-size 1 --early-stop-patience 10 \
    --save-predictions || true

# Large tier (100K-296K): transfer HP from 50K
echo "--- Large tier (transfer HP, 1 config x 3 seeds) ---"
uv run --no-sync python experiments/exp1_1_scaling.py \
    --task k562 --student legnet --oracle ag_s2 \
    --reservoir "${STRATEGY}" \
    --pool-base-dir "${POOL_DIR}" \
    --n-replicates 3 --seed 42 \
    --output-dir "${OUT_DIR}" \
    --training-sizes 100000 200000 296000 \
    --epochs 50 --ensemble-size 1 --early-stop-patience 10 \
    --transfer-hp-from 50000 \
    --save-predictions || true

echo "=== Done: $(date) ==="
