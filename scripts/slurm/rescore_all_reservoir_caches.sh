#!/bin/bash
# In-place RE-SCORE of ALL reservoir caches with the canonical full856k_clean oracle.
# Keeps sequences byte-identical; overrides stale (UNSTAMPED) labels; stamps provenance.
# 214 caches / ~42.4M sequences, LPT-balanced into 16 shards (~2.65M each, ~80 min/H100).
#   /cm/shared/apps/slurm/current/bin/sbatch --array=0-15 scripts/slurm/rescore_all_reservoir_caches.sh
#SBATCH --job-name=rescore_res
#SBATCH --output=/grid/wsbs/home_norepl/christen/ALBench-S2F/logs/%x-%A-%a.out
#SBATCH --error=/grid/wsbs/home_norepl/christen/ALBench-S2F/logs/%x-%A-%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --time=8:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G

set -euo pipefail
set +u; source /etc/profile.d/modules.sh; set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
source scripts/slurm/setup_hpc_deps.sh
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_command_buffer="
export TQDM_DISABLE=1
export PYTHONUNBUFFERED=1
export AG_ORACLE_CHUNK="${AG_ORACLE_CHUNK:-128}"

N_SHARDS="${N_SHARDS:-16}"
T="${SLURM_ARRAY_TASK_ID:-0}"

# Guard: never silently fall back to a legacy oracle.
ORACLE_DIR="outputs/oracle_full856k_clean/s2"
N_FOLDS=$(find "${ORACLE_DIR}" -maxdepth 3 -type d -path "*/best_model/checkpoint" 2>/dev/null | wc -l)
if [ "${N_FOLDS}" -lt 10 ]; then
    echo "ERROR: canonical AG_S2 oracle has only ${N_FOLDS}/10 folds in ${ORACLE_DIR} — aborting"
    exit 2
fi
echo "=== rescore_res shard=${T}/${N_SHARDS} node=${SLURMD_NODENAME} folds=${N_FOLDS} chunk=${AG_ORACLE_CHUNK} $(date) ==="

uv run --no-sync python scripts/rescore_reservoir_cache.py --shard "${T}" --n-shards "${N_SHARDS}"

echo "=== rescore_res shard ${T} DONE $(date) ==="
