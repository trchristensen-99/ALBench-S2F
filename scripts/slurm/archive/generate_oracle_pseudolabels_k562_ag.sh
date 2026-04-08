#!/bin/bash
# Generate AlphaGenome oracle ensemble pseudo-labels for K562 hashFrag splits.
#
# Produces ensemble mean/std and out-of-fold predictions for train+pool, val,
# and all three test sets (in-dist, SNV pairs, OOD).
#
# Runtime estimate:
#   - WITH test caches: ~10-20 min (all head-only, no encoder JIT needed)
#   - WITHOUT test caches: ~5-7 hours (JIT ~2h + 10 folds × ~30 min full encoder)
#
# Prerequisites:
#   1. 10 oracle checkpoints: outputs/ag_hashfrag_oracle_cached/oracle_{0-9}/best_model/
#   2. Embedding cache:       outputs/ag_hashfrag/embedding_cache/
#   3. Test set TSVs:         data/k562/test_sets/
#   4. (Optional) Test set caches: outputs/ag_hashfrag/embedding_cache/test_*.npy
#      Build with: sbatch scripts/slurm/build_hashfrag_test_embedding_cache.sh
#
#SBATCH --job-name=oracle_pseudolabels_k562_ag
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G

source /etc/profile.d/modules.sh
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD:$PYTHONPATH"
source scripts/slurm/setup_hpc_deps.sh

export XLA_FLAGS="${XLA_FLAGS} --xla_gpu_enable_command_buffer= --xla_gpu_autotune_level=0"

echo "Generating K562 AG oracle pseudolabels on $(date)"
echo "Node: ${SLURMD_NODENAME}"

uv run --no-sync python experiments/generate_oracle_pseudolabels_k562_ag.py \
    ++wandb_mode=disabled
