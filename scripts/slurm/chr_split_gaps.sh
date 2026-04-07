#!/bin/bash
# Fill chr_split gaps: AG S1/S2 extra seeds + LegNet (all cells).
#
# Gaps:
#   AG all-folds S1: 1 seed done (42), need 2 more (1042, 2042) × 3 cells = 6 runs
#   AG all-folds S2: 1 seed done (42), need 2 more (1042, 2042) × 3 cells = 6 runs
#   LegNet: 0 seeds done, need 3 (42, 1042, 2042) × 3 cells = 9 runs
#
# Array tasks:
#   0-2:  AG S1  K562/HepG2/SknSh (seeds 1042,2042)
#   3-5:  AG S2  K562/HepG2/SknSh (seeds 1042,2042)
#   6-8:  LegNet K562/HepG2/SknSh (seeds 42,1042,2042)
#
# Submit:
#   /cm/shared/apps/slurm/current/bin/sbatch --array=0-8 scripts/slurm/chr_split_gaps.sh
#
#SBATCH --job-name=chr_gaps
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=200G
#SBATCH --array=0-8

set -euo pipefail

set +u; source /etc/profile.d/modules.sh; set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
source scripts/slurm/setup_hpc_deps.sh
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_command_buffer="

T=${SLURM_ARRAY_TASK_ID}
echo "=== chr_split_gaps task=${T}  node=${SLURMD_NODENAME}  date=$(date) ==="

# Setup data symlinks for all cells
for CELL in hepg2 sknsh; do
    mkdir -p "data/${CELL}"
    ln -sf "$(pwd)/data/k562/DATA-Table_S2__MPRA_dataset.txt" "data/${CELL}/DATA-Table_S2__MPRA_dataset.txt" 2>/dev/null || true
    ln -sf "$(pwd)/data/k562/hashfrag_splits" "data/${CELL}/hashfrag_splits" 2>/dev/null || true
done

CELLS=("k562" "hepg2" "sknsh")

run_ag_s1_seeds() {
    local CELL=$1 OUT=$2
    local CELL_FLAG=""
    [[ "${CELL}" != "k562" ]] && CELL_FLAG="--cell-line ${CELL}"
    # Seed 42 already done; run 1042 and 2042
    for SEED in 1042 2042; do
        RESULT="${OUT}/genomic/n400000/hp0/seed${SEED}/result.json"
        [ -f "${RESULT}" ] && echo "  Skipping seed ${SEED} (done)" && continue
        echo "  AG S1 ${CELL} seed=${SEED}"
        uv run --no-sync python experiments/exp1_1_scaling.py \
            --task k562 --student alphagenome_k562_s1 \
            --oracle ground_truth --reservoir genomic --chr-split \
            ${CELL_FLAG} \
            --n-replicates 1 --no-hp-sweep --seed ${SEED} \
            --output-dir "${OUT}" \
            --training-sizes 400000 --epochs 50 --early-stop-patience 7 || true
    done
}

run_ag_s2_seeds() {
    local CELL=$1 OUT=$2
    local CELL_FLAG=""
    [[ "${CELL}" != "k562" ]] && CELL_FLAG="--cell-line ${CELL}"
    for SEED in 1042 2042; do
        RESULT="${OUT}/genomic/n400000/hp0/seed${SEED}/result.json"
        [ -f "${RESULT}" ] && echo "  Skipping seed ${SEED} (done)" && continue
        echo "  AG S2 ${CELL} seed=${SEED}"
        uv run --no-sync python experiments/exp1_1_scaling.py \
            --task k562 --student alphagenome_k562_s2 \
            --oracle ground_truth --reservoir genomic --chr-split \
            ${CELL_FLAG} \
            --n-replicates 1 --no-hp-sweep --seed ${SEED} \
            --output-dir "${OUT}" \
            --training-sizes 400000 --epochs 50 --early-stop-patience 7 || true
    done
}

run_legnet() {
    local CELL=$1 OUT=$2
    local CELL_FLAG=""
    [[ "${CELL}" != "k562" ]] && CELL_FLAG="--cell-line ${CELL}"
    uv run --no-sync python experiments/exp1_1_scaling.py \
        --task k562 --student legnet \
        --oracle ground_truth --reservoir genomic --chr-split \
        ${CELL_FLAG} \
        --n-replicates 3 --seed 42 \
        --output-dir "${OUT}" \
        --training-sizes 400000 --epochs 80 --ensemble-size 1 \
        --early-stop-patience 10
}

case ${T} in
    0) echo "AG S1 K562 extra seeds";  run_ag_s1_seeds k562  "outputs/chr_split/k562/ag_all_folds_s1" ;;
    1) echo "AG S1 HepG2 extra seeds"; run_ag_s1_seeds hepg2 "outputs/chr_split/hepg2/ag_all_folds_s1" ;;
    2) echo "AG S1 SknSh extra seeds"; run_ag_s1_seeds sknsh "outputs/chr_split/sknsh/ag_all_folds_s1" ;;
    3) echo "AG S2 K562 extra seeds";  run_ag_s2_seeds k562  "outputs/chr_split/k562/ag_all_folds_s2" ;;
    4) echo "AG S2 HepG2 extra seeds"; run_ag_s2_seeds hepg2 "outputs/chr_split/hepg2/ag_all_folds_s2" ;;
    5) echo "AG S2 SknSh extra seeds"; run_ag_s2_seeds sknsh "outputs/chr_split/sknsh/ag_all_folds_s2" ;;
    6) echo "LegNet K562 chr-split";   run_legnet k562  "outputs/chr_split/k562/legnet" ;;
    7) echo "LegNet HepG2 chr-split";  run_legnet hepg2 "outputs/chr_split/hepg2/legnet" ;;
    8) echo "LegNet SknSh chr-split";  run_legnet sknsh "outputs/chr_split/sknsh/legnet" ;;
    *) echo "ERROR: unknown task ${T}"; exit 1 ;;
esac

echo "=== task=${T} DONE — $(date) ==="
