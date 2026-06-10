#!/bin/bash
# #54 master-pool relabel + expansion: generate + label one (strategy, shard) per
# array task with the canonical full856k_clean AG_S2 oracle. Sequence generation is
# GPU-invariant for labeling, so this runs on idle V100s (slow_nice, 20 GPUs, 30d)
# with a smaller oracle chunk (AG_ORACLE_CHUNK) to fit the V100's 32GB.
#
# The array index -> (strategy, shard) mapping comes from
# scripts/master_pool_io.build_manifest_rows() so the .sh stays declarative.
# Pick --array=0-<N-1> where N = number of rows printed by:
#   uv run --no-sync python -c \
#     "from scripts.master_pool_io import build_manifest_rows as b; print(len(b()))"
#
# Submit (after the V100 throughput probe sets AG_ORACLE_CHUNK):
#   /cm/shared/apps/slurm/current/bin/sbatch --array=0-<N-1> \
#       scripts/slurm/relabel_master_pools.sh
#
#SBATCH --job-name=master_pool
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --time=30:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
# gres is GPU-type-agnostic on purpose: V100s are the idle tier so most tasks land
# there, but slow_nice (prio 100) also opportunistically grabs idle H100 fragments
# without preempting the H100 HP sweep. AG_ORACLE_CHUNK=32 is safe on either GPU.

set -euo pipefail
set +u; source /etc/profile.d/modules.sh; set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
source scripts/slurm/setup_hpc_deps.sh
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_command_buffer="
export TQDM_DISABLE=1
export PYTHONUNBUFFERED=1
# Smaller oracle chunk so the 551M-param backbone fits a V100 32GB (vs 128 on H100).
export AG_ORACLE_CHUNK="${AG_ORACLE_CHUNK:-32}"

T=$SLURM_ARRAY_TASK_ID

# Guard: canonical oracle must have all 10 folds before we generate anything.
ORACLE_DIR="outputs/oracle_full856k_clean/s2"
N_FOLDS=$(find "${ORACLE_DIR}" -maxdepth 3 -type d -path "*/best_model/checkpoint" 2>/dev/null | wc -l)
if [ "${N_FOLDS}" -lt 10 ]; then
    echo "ERROR: canonical AG_S2 oracle has only ${N_FOLDS}/10 folds in ${ORACLE_DIR} — aborting"
    exit 2
fi

# Resolve this array task -> strategy/mode/target/n_shards/shard.
read -r STRAT MODE TARGET NSHARDS SHARD < <(uv run --no-sync python -c "
from scripts.master_pool_io import build_manifest_rows
r = build_manifest_rows()[${T}]
print(r['strategy'], r['mode'], r['target'], r['n_shards'], r['shard'])
")

echo "=== master_pool task=${T} strat=${STRAT} mode=${MODE} shard=${SHARD}/${NSHARDS} node=${SLURMD_NODENAME} chunk=${AG_ORACLE_CHUNK} $(date) ==="

uv run --no-sync python scripts/generate_master_pool.py \
    --task k562 \
    --reservoir "${STRAT}" \
    --target "${TARGET}" \
    --n-shards "${NSHARDS}" \
    --shard "${SHARD}" \
    --mode "${MODE}" \
    --seed 42

echo "=== master_pool task=${T} DONE $(date) ==="
