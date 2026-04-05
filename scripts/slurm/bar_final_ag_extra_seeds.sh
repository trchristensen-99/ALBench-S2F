#!/bin/bash
# Add extra seeds for AG S1 and AG S2 in bar_final (chr-split, ref+alt).
# AG S1 uses full training data, AG S2 uses 20K subsample.
#
#SBATCH --job-name=ag_seeds2
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=200G
#SBATCH --time=48:00:00

set -euo pipefail
set +u; source /etc/profile.d/modules.sh; set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
source scripts/slurm/setup_hpc_deps.sh

echo "=== AG extra seeds for bar_final — $(date) ==="

for CELL in k562 hepg2 sknsh; do
    for SEED in 1042 2042; do
        # AG S1 (full dataset)
        OUT="outputs/bar_final/${CELL}/ag_s1_pred"
        if find "$OUT" -name "result.json" -path "*seed${SEED}*" 2>/dev/null | grep -q .; then
            echo "AG S1 ${CELL} s${SEED}: done"
        else
            echo "=== AG S1 | cell=$CELL | seed=$SEED — $(date) ==="
            uv run --no-sync python experiments/exp1_1_scaling.py \
                --task k562 \
                --student alphagenome_k562_s1 \
                --reservoir genomic \
                --output-dir "$OUT" \
                --training-sizes 618000 \
                --seed "$SEED" \
                --chr-split \
                --include-alt-alleles \
                --cell-line "$CELL" \
                --oracle ground_truth \
                --save-predictions || true
        fi

        # AG S2 (20K subsample, warm-start from S1)
        OUT="outputs/bar_final/${CELL}/ag_s2_rc_shift"
        if find "$OUT" -name "result.json" -path "*seed${SEED}*" 2>/dev/null | grep -q .; then
            echo "AG S2 ${CELL} s${SEED}: done"
        else
            echo "=== AG S2 | cell=$CELL | seed=$SEED — $(date) ==="
            uv run --no-sync python experiments/exp1_1_scaling.py \
                --task k562 \
                --student alphagenome_k562_s2 \
                --reservoir genomic \
                --output-dir "$OUT" \
                --training-sizes 20000 \
                --seed "$SEED" \
                --chr-split \
                --include-alt-alleles \
                --cell-line "$CELL" \
                --oracle ground_truth \
                --save-predictions || true
        fi
    done
done

echo "=== Done — $(date) ==="
