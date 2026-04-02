#!/bin/bash
# Systematic comparison: quality filters × ref+alt × augmentation × model
# All on K562 chr_split, 1 seed, for quick turnaround.
#
# The quality filters (stderr < 1.0, ±6σ outlier removal) are now built into
# K562Dataset and always applied. We compare:
#   A) Baseline (quality filtered, ref+alt, RC only)
#   B) + shift augmentation (±15bp)
#   C) + high-activity duplication (cutoff=0.5)
#   D) + shift + duplication
#
# Models: Malinois, LegNet, AG S1, AG S2 (all blocks)
# DREAM-RNN excluded per PI guidance (too slow, similar to CNN models)
#
# Array:
#   0-3:  Malinois (A, B, C, D)
#   4-7:  LegNet (A, B, C, D)
#   8-9:  AG S1 (A, C) — shift N/A for cached S1
#  10-11: AG S2 all-blocks (A, C) — with warm start from S1 in task 8
#
# Submit:
#   sbatch --array=0-7 --qos=slow_nice scripts/slurm/systematic_comparison.sh
#   sbatch --array=8-11 --qos=slow_nice scripts/slurm/systematic_comparison.sh
#
#SBATCH --job-name=sys_cmp
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
echo "=== systematic_comparison task=${T} node=${SLURMD_NODENAME} date=$(date) ==="

# Setup data symlinks
for CELL in hepg2 sknsh; do
    mkdir -p "data/${CELL}"
    ln -sf "$(pwd)/data/k562/DATA-Table_S2__MPRA_dataset.txt" "data/${CELL}/DATA-Table_S2__MPRA_dataset.txt" 2>/dev/null || true
    ln -sf "$(pwd)/data/k562/hashfrag_splits" "data/${CELL}/hashfrag_splits" 2>/dev/null || true
    ln -sf "$(pwd)/data/k562/test_sets" "data/${CELL}/test_sets" 2>/dev/null || true
done

OUT_BASE="outputs/systematic_comparison"
# Note: quality filters are now always applied in K562Dataset

# Config index within model group
MODEL_GROUP=$((T / 4))
if [ $MODEL_GROUP -ge 2 ]; then
    # AG uses 2 configs not 4 (no shift for cached S1)
    MODEL_GROUP=$(( (T - 8) / 2 + 2 ))
    CFG_IDX=$(( (T - 8) % 2 ))
else
    CFG_IDX=$((T % 4))
fi

# ── MALINOIS ──
if [ $MODEL_GROUP -eq 0 ]; then
    MODEL="malinois"
    case ${CFG_IDX} in
        0) LABEL="baseline";     SHIFT="False"; DUP="" ;;
        1) LABEL="shift";        SHIFT="True";  DUP="" ;;
        2) LABEL="dup";          SHIFT="False"; DUP="++duplication_cutoff=0.5" ;;
        3) LABEL="shift_dup";    SHIFT="True";  DUP="++duplication_cutoff=0.5" ;;
    esac
    echo "Malinois ${LABEL}"
    uv run --no-sync python experiments/train_malinois_k562.py \
        ++data_path="data/k562" \
        ++output_dir="${OUT_BASE}/${MODEL}/${LABEL}/seed_0" \
        ++seed=0 ++cell_line=k562 \
        ++chr_split=True ++include_alt_alleles=True \
        ++shift_aug="${SHIFT}" ++max_shift=15 \
        ${DUP} \
        --save-predictions 2>/dev/null || true

# ── LEGNET ──
elif [ $MODEL_GROUP -eq 1 ]; then
    MODEL="legnet"
    case ${CFG_IDX} in
        0) LABEL="baseline";     SHIFT_FLAG="" ;              DUP_FLAG="" ;;
        1) LABEL="shift";        SHIFT_FLAG="--shift-aug --max-shift 15"; DUP_FLAG="" ;;
        2) LABEL="dup";          SHIFT_FLAG="" ;              DUP_FLAG="--duplication-cutoff 0.5" ;;
        3) LABEL="shift_dup";    SHIFT_FLAG="--shift-aug --max-shift 15"; DUP_FLAG="--duplication-cutoff 0.5" ;;
    esac
    echo "LegNet ${LABEL}"
    uv run --no-sync python experiments/exp1_1_scaling.py \
        --task k562 --student legnet \
        --oracle ground_truth --reservoir genomic --chr-split \
        --include-alt-alleles \
        --n-replicates 1 --seed 42 --lr 0.005 \
        --output-dir "${OUT_BASE}/${MODEL}/${LABEL}" \
        --training-sizes 618000 --epochs 80 --ensemble-size 1 \
        --early-stop-patience 10 --save-predictions \
        ${SHIFT_FLAG} ${DUP_FLAG}

# ── AG S1 ──
elif [ $MODEL_GROUP -eq 2 ]; then
    MODEL="ag_s1"
    case ${CFG_IDX} in
        0) LABEL="baseline"; DUP_FLAG="" ;;
        1) LABEL="dup";      DUP_FLAG="--duplication-cutoff 0.5" ;;
    esac
    echo "AG S1 ${LABEL}"
    uv run --no-sync python experiments/exp1_1_scaling.py \
        --task k562 --student alphagenome_k562_s1 \
        --oracle ground_truth --reservoir genomic --chr-split \
        --include-alt-alleles \
        --n-replicates 1 --no-hp-sweep --seed 42 \
        --output-dir "${OUT_BASE}/${MODEL}/${LABEL}" \
        --training-sizes 618000 --epochs 50 --early-stop-patience 7 \
        ${DUP_FLAG}

# ── AG S2 (all blocks, warm start) ──
elif [ $MODEL_GROUP -eq 3 ]; then
    MODEL="ag_s2"
    case ${CFG_IDX} in
        0) LABEL="baseline"; DUP_FLAG="" ;;
        1) LABEL="dup";      DUP_FLAG="--duplication-cutoff 0.5" ;;
    esac
    echo "AG S2 all-blocks ${LABEL}"

    # Run S1 first if no checkpoint
    S1_CKPT="${OUT_BASE}/ag_s1/baseline/genomic/n618000/hp0/seed42/best_model/checkpoint"
    if [ ! -d "${S1_CKPT}" ]; then
        echo "  Running AG S1 first..."
        uv run --no-sync python experiments/exp1_1_scaling.py \
            --task k562 --student alphagenome_k562_s1 \
            --oracle ground_truth --reservoir genomic --chr-split \
            --include-alt-alleles \
            --n-replicates 1 --no-hp-sweep --seed 42 \
            --output-dir "${OUT_BASE}/ag_s1/baseline" \
            --training-sizes 618000 --epochs 50 --early-stop-patience 7
    fi

    uv run --no-sync python experiments/exp1_1_scaling.py \
        --task k562 --student alphagenome_k562_s2 \
        --oracle ground_truth --reservoir genomic --chr-split \
        --include-alt-alleles \
        --s1-checkpoint "${OUT_BASE}/ag_s1/baseline/genomic/n618000/hp0/seed42" \
        --n-replicates 1 --no-hp-sweep --seed 42 \
        --output-dir "${OUT_BASE}/${MODEL}/${LABEL}" \
        --training-sizes 618000 --epochs 50 --early-stop-patience 7 \
        ${DUP_FLAG}
fi

echo "=== task=${T} DONE — $(date) ==="
