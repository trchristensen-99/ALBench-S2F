#!/bin/bash
# HP search plateau analysis: find optimal search configuration.
#
# Runs 4 experiments × 3 strategies × key sizes to determine:
# 1. Trial count where more search stops helping
# 2. Which sampler works best
# 3. Whether ensembling beats more trials
# 4. Cost-saving strategies for large N
#
# Experiment × strategy matrix:
#   trial_plateau:      3 strats × 100K = 3 jobs  (~10h each: 8 trial counts × 5 seeds)
#   sampler_comparison: 3 strats × 100K = 3 jobs  (~5h each: 5 samplers × 5 seeds)
#   ensemble_depth:     3 strats × 100K = 3 jobs  (~8h each: 5 runs + cross-eval)
#   cost_mitigation:    3 strats × 200K = 3 jobs  (~8h each: 4 strategies × 3 seeds)
#
# Total: 12 jobs
# Array: experiment_idx * 3 + strat_idx
#
#SBATCH --job-name=hp_plat
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

EXPERIMENTS=("trial_plateau" "sampler_comparison" "ensemble_depth" "cost_mitigation")
STRATS=("random" "genomic" "motif_grammar")

# Sizes: trial_plateau/sampler/ensemble at 100K, cost_mitigation at 200K
SIZES=(100000 100000 100000 200000)

EXP_IDX=$((T / 3))
STRAT_IDX=$((T % 3))

EXP=${EXPERIMENTS[$EXP_IDX]}
STRAT=${STRATS[$STRAT_IDX]}
SIZE=${SIZES[$EXP_IDX]}

echo "=== HP Plateau: ${EXP} | ${STRAT} n=${SIZE} — $(date) ==="

uv run --no-sync python scripts/hp_search_plateau.py \
    --experiment "${EXP}" \
    --strategy "${STRAT}" \
    --size "${SIZE}" \
    --n-trials 20

echo "=== DONE — $(date) ==="
