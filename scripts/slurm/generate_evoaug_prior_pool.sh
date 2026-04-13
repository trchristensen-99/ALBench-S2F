#!/bin/bash
# Generate the evoaug_prior pool (2M sequences) with AG S2 oracle labels.
# Requires H100 GPU for oracle inference.
#
#SBATCH --job-name=gen_pool
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpuq
#SBATCH --qos=fast
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=200G

set -euo pipefail
set +u; source /etc/profile.d/modules.sh; set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD"
source scripts/slurm/setup_hpc_deps.sh
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_command_buffer="
export ALPHAGENOME_WEIGHTS="/grid/wsbs/home_norepl/christen/alphagenome_weights/alphagenome-jax-all_folds-v1"

echo "=== Generating evoaug_prior pool — $(date) ==="

uv run --no-sync python scripts/generate_labeled_pools.py \
    --oracle ag_s2 \
    --reservoir evoaug_prior \
    --pool-size 2000000 \
    --task k562 \
    --output-base outputs/labeled_pools_2m \
    --chr-split

echo "=== DONE — $(date) ==="
