#!/bin/bash
# Generate 3M additional evoaug_prior sequences and append to 2M pool.
# Result: 5M pool at outputs/labeled_pools_5m/k562/ag_s2/evoaug_prior/pool.npz
#
#SBATCH --job-name=gen_eap
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --time=06:00:00
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

OUT_DIR="outputs/labeled_pools_5m/k562/ag_s2/evoaug_prior"
POOL_FILE="${OUT_DIR}/pool.npz"

[ -f "${POOL_FILE}" ] && echo "SKIP: 5M pool already exists" && exit 0

echo "=== Generating evoaug_prior 5M pool — $(date) ==="

# Use the existing pool generation script which handles everything
uv run --no-sync python scripts/generate_labeled_pools.py \
    --task k562 \
    --oracle ag_s2 \
    --reservoir evoaug_prior \
    --pool-size 5000000 \
    --output-dir "${OUT_DIR}"

echo "=== DONE — $(date) ==="
