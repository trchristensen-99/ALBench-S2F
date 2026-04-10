#!/bin/bash
# Train LegNet on REAL labels at EXACTLY the same training sizes and data splits
# as the oracle-trained LegNet, for aligned Exp0 scaling comparison.
#
# Uses the same AG S2 genomic pool (618K sequences) but with ground_truth labels.
# Also trains at full ~618K and at ref+alt pool sizes if available.
#
# Submit:
#   /cm/shared/apps/slurm/current/bin/sbatch scripts/slurm/legnet_real_label_aligned.sh
#
#SBATCH --job-name=lgnt_real_align
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

set -euo pipefail
set +u; source /etc/profile.d/modules.sh; set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
source scripts/slurm/setup_hpc_deps.sh

OUT_BASE="outputs/exp0_aligned/k562/legnet_real_labels/genomic"

# Same sizes as oracle scaling
SIZES="3197 6395 15987 31974 63949 159871 296382 618000"

echo "=== LegNet real-label aligned scaling — $(date) ==="

for N in ${SIZES}; do
    # Use same HP as oracle (hp0 = default: lr=0.005, bs=1024 for small N; lr=0.001, bs=512 for large)
    for SEED in 42 1042 2042; do
        OUT="${OUT_BASE}/n${N}/hp0/seed${SEED}"
        if [ -f "${OUT}/result.json" ]; then
            echo "  n=${N} seed=${SEED}: already done"
            continue
        fi
        echo "  Training n=${N} seed=${SEED}..."
        uv run --no-sync python experiments/exp1_1_scaling.py \
            --task k562 --student legnet --oracle ground_truth \
            --reservoir genomic \
            --pool-base-dir outputs/labeled_pools/k562/ag_s2 \
            --n-replicates 1 --seed "${SEED}" \
            --output-dir "outputs/exp0_aligned/k562/legnet_real_labels" \
            --training-sizes "${N}" \
            --chr-split --lr 0.001 --batch-size 512 \
            --epochs 80 --ensemble-size 1 --early-stop-patience 10 \
            --save-predictions || true
    done
done

echo ""
echo "=== All done — $(date) ==="
