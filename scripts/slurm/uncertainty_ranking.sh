#!/bin/bash
# Rank pool sequences by oracle uncertainty (ensemble variance).
#
# One job per strategy. Each loads the 5M pool and runs 5 AG oracle folds.
# At batch_size=512 on H100: ~8 min per fold × 5 folds = ~40 min for 5M seqs.
#
# Array: 0=evoaug_heavy, 1=motif_grammar, 2=prm_1pct (from 618K pool)
#
#SBATCH --job-name=unc_rnk
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=fast
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G

set -euo pipefail
set +u; source /etc/profile.d/modules.sh; set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
source scripts/slurm/setup_hpc_deps.sh
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_command_buffer="
export ALPHAGENOME_WEIGHTS="/grid/wsbs/home_norepl/christen/alphagenome_weights/alphagenome-jax-all_folds-v1"

T=$SLURM_ARRAY_TASK_ID

STRATS=("evoaug_heavy" "motif_grammar" "prm_1pct")
POOL_SIZES=("5m" "5m" "618k")

STRAT=${STRATS[$T]}
POOL_SIZE=${POOL_SIZES[$T]}

echo "=== Uncertainty Ranking: ${STRAT} (${POOL_SIZE}) — $(date) ==="

uv run --no-sync python scripts/rank_pool_by_uncertainty.py \
    --strategy "${STRAT}" \
    --pool-size "${POOL_SIZE}" \
    --n-folds 5

echo "=== DONE — $(date) ==="
