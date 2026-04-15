#!/bin/bash
# Lean HP search comparison: fill gaps in existing knowledge.
#
# 4 experiments × 3 strategies × 2 sizes (50K, 100K) = 24 jobs
#
# Experiments:
#   0: trial_extension — 30 and 40 trials (we have 10 and 20 already)
#   1: sampler_alt     — GP, QMC, CMA-ES vs TPE baseline (20 trials)
#   2: cost_proxy      — 15ep proxy, subsample, combined
#   3: ensemble_lean   — depth 1 vs 3
#
# Array: exp_idx * 6 + strat_idx * 2 + size_idx
# Total: 4 × 3 × 2 = 24 jobs
#
# Time estimates:
#   trial_extension: ~4h (30+40 trials × 3 seeds)
#   sampler_alt:     ~4h (4 samplers × 20 trials × 3 seeds)
#   cost_proxy:      ~6h (4 methods × 20 trials × 3 seeds + 12 full evals)
#   ensemble_lean:   ~5h (3 searches + cross-eval + full evals)
#
#SBATCH --job-name=hp_lean
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

EXPERIMENTS=("trial_extension" "sampler_alt" "cost_proxy" "ensemble_lean")
STRATS=("random" "genomic" "motif_grammar")
SIZES=(50000 100000)

EXP_IDX=$((T / 6))
STRAT_IDX=$(( (T % 6) / 2 ))
SIZE_IDX=$((T % 2))

EXP=${EXPERIMENTS[$EXP_IDX]}
STRAT=${STRATS[$STRAT_IDX]}
SIZE=${SIZES[$SIZE_IDX]}

echo "=== HP Lean: ${EXP} | ${STRAT} n=${SIZE} — $(date) ==="

uv run --no-sync python scripts/hp_search_lean.py \
    --experiment "${EXP}" \
    --strategy "${STRAT}" \
    --size "${SIZE}"

echo "=== DONE — $(date) ==="
