#!/bin/bash
# Build embedding cache for K562 hashFrag test sets (in-dist, SNV, OOD).
#
# One-time cost (~30-60 min after JIT compilation).  Subsequent pseudolabel
# generation can use head-only inference for ALL splits, eliminating the
# 2h+ JIT + 30 min/fold encoder overhead.
#
# Submit:
#   /cm/shared/apps/slurm/current/bin/sbatch scripts/slurm/build_hashfrag_test_embedding_cache.sh
#
#SBATCH --job-name=build_hf_test_cache
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G

set -euo pipefail

set +u
source /etc/profile.d/modules.sh
set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
source scripts/slurm/setup_hpc_deps.sh

export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_command_buffer="

echo "Building hashFrag test set embedding caches on $(date)"
echo "Node: ${SLURMD_NODENAME}"

WEIGHTS=/grid/wsbs/home_norepl/christen/alphagenome_weights/alphagenome-jax-all_folds-v1

uv run --no-sync python scripts/build_test_embedding_cache.py \
    --cache-dir outputs/ag_hashfrag/embedding_cache \
    --k562-data-path data/k562 \
    --weights-path "$WEIGHTS" \
    --batch-size 256

echo "Done — $(date)"
