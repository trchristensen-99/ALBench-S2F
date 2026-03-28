#!/bin/bash
# Train all models on chromosome-based splits for K562, HepG2, and SK-N-SH.
# Chr split: test=chr7+13, val=chr19+21+X, train=rest.
# All models use real labels (ground_truth oracle, genomic reservoir).
# All from-scratch models use ensemble_size=1 for fair comparison.
#
# Array tasks (K562):
#   0 = DREAM-RNN K562 (3 seeds)
#   1 = DREAM-CNN K562 (3 seeds)
#   2 = AG fold-1 S1 K562
#   3 = AG all-folds S1 K562
#   4 = AG all-folds S2 K562
#   13 = Malinois K562 (3 seeds)
# Array tasks (HepG2):
#   5 = DREAM-RNN HepG2 (3 seeds)
#   6 = AG fold-1 S1 HepG2
#   7 = AG all-folds S1 HepG2
#   8 = AG all-folds S2 HepG2
#   14 = Malinois HepG2 (3 seeds)
# Array tasks (SK-N-SH):
#   9 = DREAM-RNN SKNSH (3 seeds)
#   10 = AG fold-1 S1 SKNSH
#   11 = AG all-folds S1 SKNSH
#   12 = AG all-folds S2 SKNSH
#   15 = Malinois SKNSH (3 seeds)
#
# NOTE: Foundation models (Enformer/Borzoi/NTv3) on chr-split need
# cache re-indexing — handled in separate scripts.
#
# Submit:
#   sbatch --array=0-12 scripts/slurm/train_chr_split_all.sh
#
#SBATCH --job-name=chr_split
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=200G
#SBATCH --array=0-15

set -euo pipefail

set +u; source /etc/profile.d/modules.sh; set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
source scripts/slurm/setup_hpc_deps.sh
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_command_buffer="

T=${SLURM_ARRAY_TASK_ID}
echo "=== chr_split task=${T}  node=${SLURMD_NODENAME}  date=$(date) ==="

# Setup data symlinks for all cells
for CELL in hepg2 sknsh; do
    mkdir -p "data/${CELL}"
    ln -sf "$(pwd)/data/k562/DATA-Table_S2__MPRA_dataset.txt" "data/${CELL}/DATA-Table_S2__MPRA_dataset.txt" 2>/dev/null || true
    ln -sf "$(pwd)/data/k562/hashfrag_splits" "data/${CELL}/hashfrag_splits" 2>/dev/null || true
done

# Helper function for DREAM models
run_dream() {
    local STUDENT=$1 CELL=$2 OUT=$3
    local CELL_FLAG=""
    [[ "${CELL}" != "k562" ]] && CELL_FLAG="--cell-line ${CELL}"
    uv run --no-sync python experiments/exp1_1_scaling.py \
        --task k562 --student "${STUDENT}" \
        --oracle ground_truth --reservoir genomic --chr-split \
        ${CELL_FLAG} \
        --n-replicates 3 --seed 42 \
        --output-dir "${OUT}" \
        --training-sizes 400000 --epochs 80 --ensemble-size 1 \
        --early-stop-patience 10
}

# Helper for AG S1
run_ag_s1() {
    local WEIGHTS=$1 CELL=$2 OUT=$3
    local CELL_FLAG=""
    [[ "${CELL}" != "k562" ]] && CELL_FLAG="--cell-line ${CELL}"
    [[ -n "${WEIGHTS}" ]] && export ALPHAGENOME_WEIGHTS="${WEIGHTS}"
    uv run --no-sync python experiments/exp1_1_scaling.py \
        --task k562 --student alphagenome_k562_s1 \
        --oracle ground_truth --reservoir genomic --chr-split \
        ${CELL_FLAG} \
        --n-replicates 1 --no-hp-sweep --seed 42 \
        --output-dir "${OUT}" \
        --training-sizes 400000 --epochs 50 --early-stop-patience 7
}

# Helper for AG S2
run_ag_s2() {
    local CELL=$1 OUT=$2
    local CELL_FLAG=""
    [[ "${CELL}" != "k562" ]] && CELL_FLAG="--cell-line ${CELL}"
    uv run --no-sync python experiments/exp1_1_scaling.py \
        --task k562 --student alphagenome_k562_s2 \
        --oracle ground_truth --reservoir genomic --chr-split \
        ${CELL_FLAG} \
        --n-replicates 1 --no-hp-sweep --seed 42 \
        --output-dir "${OUT}" \
        --training-sizes 400000 --epochs 50 --early-stop-patience 7
}

FOLD1="/grid/wsbs/home_norepl/christen/alphagenome_weights/alphagenome-jax-fold_1"

# Helper for Malinois
run_malinois() {
    local CELL=$1 OUT=$2
    for SEED in 0 1 2; do
        echo "--- Malinois ${CELL} seed ${SEED} ---"
        uv run --no-sync python experiments/train_malinois_k562.py \
            ++data_path="data/${CELL}" \
            ++output_dir="${OUT}/seed_${SEED}" \
            ++seed="${SEED}" \
            ++cell_line="${CELL}" \
            ++chr_split=True
    done
}

case ${T} in
    # --- K562 ---
    0)  echo "DREAM-RNN K562 chr-split"; run_dream dream_rnn k562 "outputs/chr_split/k562/dream_rnn" ;;
    1)  echo "DREAM-CNN K562 chr-split"; run_dream dream_cnn k562 "outputs/chr_split/k562/dream_cnn" ;;
    2)  echo "AG fold-1 S1 K562 chr-split"; run_ag_s1 "$FOLD1" k562 "outputs/chr_split/k562/ag_fold_1_s1" ;;
    3)  echo "AG all-folds S1 K562 chr-split"; run_ag_s1 "" k562 "outputs/chr_split/k562/ag_all_folds_s1" ;;
    4)  echo "AG all-folds S2 K562 chr-split"; run_ag_s2 k562 "outputs/chr_split/k562/ag_all_folds_s2" ;;
    # --- HepG2 ---
    5)  echo "DREAM-RNN HepG2 chr-split"; run_dream dream_rnn hepg2 "outputs/chr_split/hepg2/dream_rnn" ;;
    6)  echo "AG fold-1 S1 HepG2 chr-split"; run_ag_s1 "$FOLD1" hepg2 "outputs/chr_split/hepg2/ag_fold_1_s1" ;;
    7)  echo "AG all-folds S1 HepG2 chr-split"; run_ag_s1 "" hepg2 "outputs/chr_split/hepg2/ag_all_folds_s1" ;;
    8)  echo "AG all-folds S2 HepG2 chr-split"; run_ag_s2 hepg2 "outputs/chr_split/hepg2/ag_all_folds_s2" ;;
    # --- SK-N-SH ---
    9)  echo "DREAM-RNN SKNSH chr-split"; run_dream dream_rnn sknsh "outputs/chr_split/sknsh/dream_rnn" ;;
    10) echo "AG fold-1 S1 SKNSH chr-split"; run_ag_s1 "$FOLD1" sknsh "outputs/chr_split/sknsh/ag_fold_1_s1" ;;
    11) echo "AG all-folds S1 SKNSH chr-split"; run_ag_s1 "" sknsh "outputs/chr_split/sknsh/ag_all_folds_s1" ;;
    12) echo "AG all-folds S2 SKNSH chr-split"; run_ag_s2 sknsh "outputs/chr_split/sknsh/ag_all_folds_s2" ;;
    # --- Malinois ---
    13) echo "Malinois K562 chr-split"; run_malinois k562 "outputs/chr_split/k562/malinois" ;;
    14) echo "Malinois HepG2 chr-split"; run_malinois hepg2 "outputs/chr_split/hepg2/malinois" ;;
    15) echo "Malinois SKNSH chr-split"; run_malinois sknsh "outputs/chr_split/sknsh/malinois" ;;
    *)  echo "ERROR: unknown task ${T}"; exit 1 ;;
esac

echo "=== task=${T} DONE — $(date) ==="
