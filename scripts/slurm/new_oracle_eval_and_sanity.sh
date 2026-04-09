#!/bin/bash
# Generate test labels with NEW oracle + quick scaling sanity check
#
#SBATCH --job-name=new_orc_eval
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpuq
#SBATCH --qos=default
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=200G

set -euo pipefail

set +u; source /etc/profile.d/modules.sh; set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
source scripts/slurm/setup_hpc_deps.sh
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_command_buffer="
export ALPHAGENOME_WEIGHTS="/grid/wsbs/home_norepl/christen/alphagenome_weights/alphagenome-jax-all_folds-v1"

echo "=== Step 1: Generate chr-split test labels with NEW oracle — $(date) ==="
uv run --no-sync python scripts/generate_ag_s2_test_labels.py

echo ""
echo "=== Step 2: Re-label genomic + random pools — $(date) ==="
for STRAT in genomic random; do
    echo "  Re-generating ${STRAT}..."
    rm -f "outputs/labeled_pools/k562/ag_s2/${STRAT}/pool.npz"
    uv run --no-sync python scripts/generate_labeled_pools.py \
        --task k562 --oracle ag_s2 \
        --reservoir "${STRAT}" \
        --pool-size 618000 \
        --chr-split --include-alt-alleles --seed 42
done

echo ""
echo "=== Step 3: Quick scaling sanity check (3 sizes, 1 seed) — $(date) ==="
for STRAT in genomic random; do
    echo "  ${STRAT}..."
    uv run --no-sync python experiments/exp1_1_scaling.py \
        --task k562 --student legnet --oracle ag_s2 \
        --reservoir "${STRAT}" \
        --pool-base-dir outputs/labeled_pools/k562/ag_s2 \
        --n-replicates 1 --seed 42 \
        --output-dir outputs/exp1_1_new_oracle/k562/legnet_ag_s2 \
        --training-sizes 5000 50000 200000 \
        --chr-split --lr 0.001 --batch-size 1024 \
        --epochs 80 --ensemble-size 1 --early-stop-patience 10 \
        --save-predictions || true
done

echo ""
echo "=== Step 4: Run oracle landscape analysis — $(date) ==="
uv run --no-sync python scripts/analysis/analyze_oracle_landscapes.py || true

echo ""
echo "=== All done — $(date) ==="
