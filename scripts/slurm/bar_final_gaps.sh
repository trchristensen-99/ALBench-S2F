#!/bin/bash
# Fill gaps in bar_final results: missing seeds, DREAM-CNN, predictions.
#
# This handles from-scratch PyTorch models (DREAM-RNN, DREAM-CNN, LegNet)
# across all 3 cell types with 3 seeds, saving predictions for scatter plots.
#
# Submit:
#   /cm/shared/apps/slurm/current/bin/sbatch scripts/slurm/bar_final_gaps.sh
#
#SBATCH --job-name=bar_gaps
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00

set -euo pipefail
set +u; source /etc/profile.d/modules.sh; set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
source scripts/slurm/setup_hpc_deps.sh

echo "=== Bar_final gap filler — $(date) ==="
echo "Node: ${SLURMD_NODENAME}"

# From-scratch models via exp1_1_scaling.py
# These use chr_split, include_alt_alleles, quality filters, save_predictions
run_from_scratch() {
    local STUDENT=$1
    local CELL=$2
    local SEED=$3
    local OUT_BASE="outputs/bar_final/${CELL}/${STUDENT}"
    local RESULT

    # Check for result in various possible paths
    for subdir in "genomic/n618000/hp0/seed${SEED}" "genomic/n400000/hp0/seed${SEED}"; do
        RESULT="${OUT_BASE}/${subdir}/result.json"
        if [ -f "$RESULT" ]; then
            echo "  ${STUDENT}/${CELL}/s${SEED}: done"
            return 0
        fi
    done

    echo "  ${STUDENT}/${CELL}/s${SEED} — $(date)"
    uv run --no-sync python experiments/exp1_1_scaling.py \
        --task "k562" \
        --student "${STUDENT}" \
        --reservoir genomic \
        --output-dir "${OUT_BASE}" \
        --training-sizes 618000 \
        --seed "${SEED}" \
        --chr-split \
        --include-alt-alleles \
        --save-predictions \
        --cell-line "${CELL}" \
        --ensemble-size 1
}

# Run from-scratch models for all cell types and seeds
for STUDENT in dream_rnn dream_cnn legnet; do
    echo ""
    echo "--- ${STUDENT} ---"
    for CELL in k562 hepg2 sknsh; do
        for SEED in 42 1042 2042; do
            run_from_scratch "${STUDENT}" "${CELL}" "${SEED}" || true
        done
    done
done

# Ensemble runs (3-model ensemble for bar plot comparison)
echo ""
echo "--- Ensembles ---"
for STUDENT in dream_rnn legnet; do
    for CELL in k562 hepg2 sknsh; do
        OUT="outputs/bar_final/${CELL}/${STUDENT}_ens3"
        FOUND=0
        for subdir in "genomic/n618000/hp0/seed42" "genomic/n400000/hp0/seed42"; do
            if [ -f "${OUT}/${subdir}/result.json" ]; then
                echo "  ${STUDENT}_ens3/${CELL}: done"
                FOUND=1
                break
            fi
        done
        [ "$FOUND" -eq 1 ] && continue

        echo "  ${STUDENT}_ens3/${CELL} — $(date)"
        uv run --no-sync python experiments/exp1_1_scaling.py \
            --task "k562" \
            --student "${STUDENT}" \
            --reservoir genomic \
            --output-dir "${OUT}" \
            --training-sizes 618000 \
            --seed 42 \
            --chr-split \
            --include-alt-alleles \
            \
            --save-predictions \
            --cell-line "${CELL}" \
            --ensemble-size 3 || true
    done
done

echo ""
echo "=== Bar_final gaps DONE — $(date) ==="
