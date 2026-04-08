#!/bin/bash
# LegNet architecture sweep: narrow/default/wide × lr × bs on cached pools.
# Tests whether model capacity should scale with data size.
#
# Peter's note: "number of layers and width as well as amount of reg.
# Smaller datasets will need more [regularization]."
#
# Architectures:
#   narrow: [128,128,64,64,32,32,16,16] (~500K params)
#   default: [256,256,128,128,64,64,32,32] (~2.6M params)
#   wide: [512,512,256,256,128,128,64,64] (~10M params)
#
# HP grid: 3 archs × 2 lr × 2 bs = 12 configs at small N
#          3 archs × 1 lr × 1 bs = 3 configs at large N
#
# Array: one task per priority strategy (genomic, random, prm_5pct, evoaug)
#   0: genomic   1: random   2: prm_5pct   3: evoaug_structural
#
# Submit:
#   /cm/shared/apps/slurm/current/bin/sbatch --array=0-3 scripts/slurm/exp1_1_legnet_arch_sweep.sh
#
#SBATCH --job-name=lgnt_arch
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
STRATEGIES=(genomic random prm_5pct evoaug_structural)
STRATEGY="${STRATEGIES[$T]}"
POOL_DIR="outputs/labeled_pools/k562/ag_s2"
OUT_DIR="outputs/exp1_1_arch_sweep/k562/legnet_ag_s2"

echo "=== LegNet arch sweep: ${STRATEGY} node=${SLURMD_NODENAME} $(date) ==="

# Small tier (1K-50K): full arch+HP sweep (12 configs × 3 seeds = 36 per size)
echo "--- Small tier (arch sweep, 12 configs × 3 seeds) ---"
uv run --no-sync python experiments/exp1_1_scaling.py \
    --task k562 --student legnet --oracle ag_s2 \
    --reservoir "${STRATEGY}" \
    --pool-base-dir "${POOL_DIR}" \
    --n-replicates 3 --seed 42 \
    --output-dir "${OUT_DIR}" \
    --training-sizes 1000 5000 10000 20000 50000 \
    --chr-split --arch-sweep \
    --epochs 80 --ensemble-size 1 --early-stop-patience 10 \
    --save-predictions || true

# Large tier (100K-500K): transfer best arch+HP from 50K
echo "--- Large tier (transfer best arch+HP, 3 seeds) ---"
uv run --no-sync python experiments/exp1_1_scaling.py \
    --task k562 --student legnet --oracle ag_s2 \
    --reservoir "${STRATEGY}" \
    --pool-base-dir "${POOL_DIR}" \
    --n-replicates 3 --seed 42 \
    --output-dir "${OUT_DIR}" \
    --training-sizes 100000 200000 500000 \
    --chr-split --arch-sweep \
    --epochs 50 --ensemble-size 1 --early-stop-patience 10 \
    --transfer-hp-from 50000 \
    --save-predictions || true

echo "=== Done: $(date) ==="
