#!/bin/bash
# Config B: chr_split ref+alt (train~618K, test=chr7+13 ~62K)
# Same models as Config A but with ref+alt data.
#
#SBATCH --job-name=bar_alt
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

echo "=== Config B: chr_split ref+alt — $(date) ==="

for CELL in k562 hepg2 sknsh; do
    echo "--- Cell: ${CELL} ---"

    # AG S1
    for SEED in 42 1042 2042; do
        OUT="outputs/bar_std_refalt/${CELL}/ag_s1"
        [ -f "${OUT}/genomic/n618000/hp0/seed${SEED}/result.json" ] && continue
        echo "  AG S1 ${CELL} s${SEED}"
        uv run --no-sync python experiments/exp1_1_scaling.py \
            --task k562 --student alphagenome_k562_s1 --reservoir genomic \
            --output-dir "${OUT}" --training-sizes 618000 --seed ${SEED} \
            --chr-split --include-alt-alleles --cell-line ${CELL} \
            --oracle ground_truth --save-predictions --ensemble-size 1 || true
    done

    # AG S2
    for SEED in 42 1042 2042; do
        OUT="outputs/bar_std_refalt/${CELL}/ag_s2"
        [ -f "${OUT}/genomic/n20000/hp0/seed${SEED}/result.json" ] && continue
        echo "  AG S2 ${CELL} s${SEED}"
        uv run --no-sync python experiments/exp1_1_scaling.py \
            --task k562 --student alphagenome_k562_s2 --reservoir genomic \
            --output-dir "${OUT}" --training-sizes 20000 --seed ${SEED} \
            --chr-split --include-alt-alleles --cell-line ${CELL} \
            --oracle ground_truth --save-predictions || true
    done

    # DREAM-RNN
    for SEED in 42 1042 2042; do
        OUT="outputs/bar_std_refalt/${CELL}/dream_rnn"
        [ -f "${OUT}/genomic/n618000/hp0/seed${SEED}/result.json" ] && continue
        echo "  DREAM-RNN ${CELL} s${SEED}"
        uv run --no-sync python experiments/exp1_1_scaling.py \
            --task k562 --student dream_rnn --reservoir genomic \
            --output-dir "${OUT}" --training-sizes 618000 --seed ${SEED} \
            --chr-split --include-alt-alleles --cell-line ${CELL} \
            --oracle ground_truth --save-predictions --ensemble-size 1 || true
    done

    # LegNet
    for SEED in 42 1042 2042; do
        OUT="outputs/bar_std_refalt/${CELL}/legnet"
        [ -f "${OUT}/genomic/n618000/hp0/seed${SEED}/result.json" ] && continue
        echo "  LegNet ${CELL} s${SEED}"
        uv run --no-sync python experiments/exp1_1_scaling.py \
            --task k562 --student legnet --reservoir genomic \
            --output-dir "${OUT}" --training-sizes 618000 --seed ${SEED} \
            --chr-split --include-alt-alleles --cell-line ${CELL} \
            --oracle ground_truth --save-predictions --ensemble-size 1 || true
    done

    # DREAM-CNN
    for SEED in 42 1042 2042; do
        OUT="outputs/bar_std_refalt/${CELL}/dream_cnn"
        [ -f "${OUT}/genomic/n618000/hp0/seed${SEED}/result.json" ] && continue
        echo "  DREAM-CNN ${CELL} s${SEED}"
        uv run --no-sync python experiments/exp1_1_scaling.py \
            --task k562 --student dream_cnn --reservoir genomic \
            --output-dir "${OUT}" --training-sizes 618000 --seed ${SEED} \
            --chr-split --include-alt-alleles --cell-line ${CELL} \
            --oracle ground_truth --save-predictions --ensemble-size 1 || true
    done
done

# Malinois ref+alt with shift+dup
for CELL in k562 hepg2 sknsh; do
    for SEED in 0 1 2; do
        OUT="outputs/bar_std_refalt/${CELL}/malinois"
        [ -f "${OUT}/seed_${SEED}/result.json" ] && continue
        echo "  Malinois ${CELL} s${SEED}"
        uv run --no-sync python experiments/train_malinois_k562.py \
            ++output_dir="${OUT}" ++seed=${SEED} \
            ++chr_split=True ++include_alt_alleles=True \
            ++shift_aug=True ++max_shift=15 ++duplication_cutoff=0.5 \
            ++cell_line=${CELL} || true
    done
done

echo "=== Config B DONE — $(date) ==="
