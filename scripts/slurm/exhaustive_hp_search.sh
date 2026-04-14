#!/bin/bash
# Exhaustive HP search method comparison.
#
# Tests 14 Optuna sampler configurations × 3 strategies × 3 sizes = 126 jobs.
# Each job runs 3 iterations (different data subsamples) internally and
# measures HP consistency + cross-evaluation performance.
#
# Methods (14):
#   0:  tpe_10_cold         — TPE, 10 trials, no warm-start (baseline)
#   1:  tpe_20_cold         — TPE, 20 trials, no warm-start
#   2:  tpe_20_warm         — TPE, 20 trials, warm-start
#   3:  tpe_30_warm         — TPE, 30 trials, warm-start
#   4:  tpe_50_warm         — TPE, 50 trials, warm-start (exhaustive)
#   5:  tpe_20_multi_warm   — TPE multivariate, 20 trials, warm-start
#   6:  tpe_20_warm_narrow  — TPE, 20 trials, warm-start, narrow LR [5e-4,1e-2]
#   7:  cma_20_warm         — CMA-ES, 20 trials, warm-start
#   8:  cma_30_warm         — CMA-ES, 30 trials, warm-start
#   9:  random_20_warm      — Random, 20 trials, warm-start
#   10: random_50_warm      — Random, 50 trials, warm-start
#   11: gp_20_warm          — GP-based, 20 trials, warm-start
#   12: ensemble_3x20       — 3×TPE-20-warm, cross-eval, consensus
#   13: (reserved)
#
# Strategies: random, genomic, motif_grammar
# Sizes: 50000, 100000, 200000
#
# Array indexing: method_idx * 9 + strat_idx * 3 + size_idx
# Total: 13 methods × 3 strategies × 3 sizes = 117 jobs
# Array: 0-116
#
# Wall time estimates per job (3 iterations × n_trials × ~3min/trial):
#   10 trials:  ~1.5h    30 trials:  ~4.5h
#   20 trials:  ~3h      50 trials:  ~7.5h
#   ensemble:   ~8h (3×20 + 9×3 cross-eval)
#
#SBATCH --job-name=xhp_srch
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

set -euo pipefail
set +u; source /etc/profile.d/modules.sh; set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD"
export TORCHDYNAMO_DISABLE=1
source scripts/slurm/setup_hpc_deps.sh

T=$SLURM_ARRAY_TASK_ID

METHODS=(
    "tpe_10_cold"
    "tpe_20_cold"
    "tpe_20_warm"
    "tpe_30_warm"
    "tpe_50_warm"
    "tpe_20_multi_warm"
    "tpe_20_warm_narrow"
    "cma_20_warm"
    "cma_30_warm"
    "random_20_warm"
    "random_50_warm"
    "gp_20_warm"
    "ensemble_3x20"
)
STRATS=("random" "genomic" "motif_grammar")
SIZES=(50000 100000 200000)

N_METHODS=${#METHODS[@]}
N_STRATS=${#STRATS[@]}
N_SIZES=${#SIZES[@]}

METHOD_IDX=$((T / (N_STRATS * N_SIZES)))
STRAT_IDX=$(( (T / N_SIZES) % N_STRATS ))
SIZE_IDX=$((T % N_SIZES))

# Bounds check
if [ $METHOD_IDX -ge $N_METHODS ]; then
    echo "Task $T out of range (method_idx=$METHOD_IDX >= $N_METHODS)"
    exit 0
fi

METHOD=${METHODS[$METHOD_IDX]}
STRAT=${STRATS[$STRAT_IDX]}
SIZE=${SIZES[$SIZE_IDX]}

echo "=== Exhaustive HP: ${METHOD} | ${STRAT} n=${SIZE} — $(date) ==="

uv run --no-sync python scripts/exhaustive_hp_search.py \
    --method "${METHOD}" \
    --strategy "${STRAT}" \
    --size "${SIZE}"

echo "=== DONE — $(date) ==="
