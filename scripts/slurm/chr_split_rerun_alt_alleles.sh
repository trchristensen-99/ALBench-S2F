#!/bin/bash
# Rerun ALL chr_split models with include_alt_alleles=True to match Malinois paper.
# Previous chr_split runs used ref-only (~336K train, ~33K test).
# Paper uses ref+alt (~659K train, ~66K test).
#
# Array:
#   0 = DREAM-RNN K562 (3 seeds)
#   1 = Malinois K562 (3 seeds)
#   2 = AG all-folds S1 K562
#   3 = AG all-folds S2 K562
#   4 = DREAM-RNN HepG2 (3 seeds)
#   5 = Malinois HepG2 (3 seeds)
#   6 = AG all-folds S1 HepG2
#   7 = AG all-folds S2 HepG2
#   8 = DREAM-RNN SKNSH (3 seeds)
#   9 = Malinois SKNSH (3 seeds)
#  10 = AG all-folds S1 SKNSH
#  11 = AG all-folds S2 SKNSH
#
# Submit across QoS tiers:
#   sbatch --array=0-3 --qos=default --time=12:00:00 scripts/slurm/chr_split_rerun_alt_alleles.sh
#   sbatch --array=4-11 --qos=slow_nice scripts/slurm/chr_split_rerun_alt_alleles.sh
#
#SBATCH --job-name=chr_alt
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --time=48:00:00
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

T=$SLURM_ARRAY_TASK_ID
echo "=== chr_split_rerun_alt task=${T} node=${SLURMD_NODENAME} date=$(date) ==="

# Setup data symlinks for all cells
for CELL in hepg2 sknsh; do
    mkdir -p "data/${CELL}"
    ln -sf "$(pwd)/data/k562/DATA-Table_S2__MPRA_dataset.txt" "data/${CELL}/DATA-Table_S2__MPRA_dataset.txt" 2>/dev/null || true
    ln -sf "$(pwd)/data/k562/hashfrag_splits" "data/${CELL}/hashfrag_splits" 2>/dev/null || true
    ln -sf "$(pwd)/data/k562/test_sets" "data/${CELL}/test_sets" 2>/dev/null || true
done

# Map task ID to (model, cell)
CELLS=("k562" "k562" "k562" "k562" "hepg2" "hepg2" "hepg2" "hepg2" "sknsh" "sknsh" "sknsh" "sknsh")
CELL="${CELLS[$T]}"
MODEL_IDX=$((T % 4))

OUT_BASE="outputs/chr_split_v2/${CELL}"

# Helper: DREAM-RNN
run_dream() {
    local CELL=$1 OUT=$2
    local CELL_FLAG=""
    [[ "${CELL}" != "k562" ]] && CELL_FLAG="--cell-line ${CELL}"
    uv run --no-sync python experiments/exp1_1_scaling.py \
        --task k562 --student dream_rnn \
        --oracle ground_truth --reservoir genomic --chr-split \
        --include-alt-alleles \
        ${CELL_FLAG} \
        --n-replicates 3 --seed 42 \
        --output-dir "${OUT}" \
        --training-sizes 700000 --epochs 80 --ensemble-size 1 \
        --early-stop-patience 10
}

# Helper: Malinois
run_malinois() {
    local CELL=$1 OUT=$2
    for SEED in 0 1 2; do
        echo "--- Malinois ${CELL} seed ${SEED} ---"
        uv run --no-sync python experiments/train_malinois_k562.py \
            ++data_path="data/${CELL}" \
            ++output_dir="${OUT}/seed_${SEED}" \
            ++seed="${SEED}" \
            ++cell_line="${CELL}" \
            ++chr_split=True \
            ++include_alt_alleles=True
    done
}

# Helper: AG S1
run_ag_s1() {
    local CELL=$1 OUT=$2
    local CELL_FLAG=""
    [[ "${CELL}" != "k562" ]] && CELL_FLAG="--cell-line ${CELL}"
    uv run --no-sync python experiments/exp1_1_scaling.py \
        --task k562 --student alphagenome_k562_s1 \
        --oracle ground_truth --reservoir genomic --chr-split \
        --include-alt-alleles \
        ${CELL_FLAG} \
        --n-replicates 1 --no-hp-sweep --seed 42 \
        --output-dir "${OUT}" \
        --training-sizes 700000 --epochs 50 --early-stop-patience 7
}

# Helper: AG S2 (requires S1 checkpoint for warm start)
run_ag_s2() {
    local CELL=$1 OUT=$2
    local CELL_FLAG=""
    [[ "${CELL}" != "k562" ]] && CELL_FLAG="--cell-line ${CELL}"

    # Find best S1 checkpoint for warm start
    S1_DIR="outputs/chr_split_v2/${CELL}/ag_all_folds_s1"
    S1_CKPT=""
    # Search for S1 checkpoint in standard locations
    for CANDIDATE in \
        "${S1_DIR}/genomic/n700000/hp0/seed42/best_model/checkpoint" \
        "${S1_DIR}/genomic/n700000/hp0/seed42" \
        "outputs/chr_split/${CELL}/ag_all_folds_s1/genomic/n400000/hp0/seed42"; do
        if [ -d "${CANDIDATE}" ]; then
            S1_CKPT="${CANDIDATE}"
            break
        fi
    done

    S1_FLAG=""
    if [ -n "${S1_CKPT}" ]; then
        echo "  S2 warm start from S1: ${S1_CKPT}"
        S1_FLAG="--s1-checkpoint ${S1_CKPT}"
    else
        echo "  WARNING: No S1 checkpoint found; S2 will cold start"
    fi

    uv run --no-sync python experiments/exp1_1_scaling.py \
        --task k562 --student alphagenome_k562_s2 \
        --oracle ground_truth --reservoir genomic --chr-split \
        --include-alt-alleles \
        ${CELL_FLAG} \
        --n-replicates 1 --no-hp-sweep --seed 42 \
        --output-dir "${OUT}" \
        --training-sizes 700000 --epochs 50 --early-stop-patience 7 \
        ${S1_FLAG}
}

case ${MODEL_IDX} in
    0) echo "DREAM-RNN ${CELL}"; run_dream "${CELL}" "${OUT_BASE}/dream_rnn" ;;
    1) echo "Malinois ${CELL}";   run_malinois "${CELL}" "${OUT_BASE}/malinois" ;;
    2) echo "AG S1 ${CELL}";      run_ag_s1 "${CELL}" "${OUT_BASE}/ag_all_folds_s1" ;;
    3) echo "AG S2 ${CELL}";
       # S2 depends on S1 — run S1 first if checkpoint doesn't exist
       S1_OUT="${OUT_BASE}/ag_all_folds_s1"
       S1_CKPT_CHECK="${S1_OUT}/genomic/n700000/hp0/seed42/best_model/checkpoint"
       if [ ! -d "${S1_CKPT_CHECK}" ]; then
           echo "  Running AG S1 first (no existing checkpoint)..."
           run_ag_s1 "${CELL}" "${S1_OUT}"
       fi
       run_ag_s2 "${CELL}" "${OUT_BASE}/ag_all_folds_s2" ;;
esac

echo "=== task=${T} DONE — $(date) ==="
