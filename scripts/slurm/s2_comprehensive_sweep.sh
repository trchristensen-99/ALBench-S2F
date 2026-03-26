#!/bin/bash
# Comprehensive S2 HP sweep for ALL foundation models on HepG2/SK-N-SH.
# Tests multiple encoder_lr and unfreeze strategies to find optimal S2
# config per model per cell line.
#
# AG configs: vary encoder_lr AND unfreeze depth
# NTv3 configs: vary encoder_lr (already uses unfreeze=all)
# Enformer: already being swept in enformer_s2_multicell_sweep.sh
#
# Array:
#   --- AG all-folds ---
#   0:  HepG2, enc_lr=1e-4, unfreeze=[4,5] (K562 best)
#   1:  SKNSH, enc_lr=1e-4, unfreeze=[4,5]
#   2:  HepG2, enc_lr=5e-5, unfreeze=[4,5]
#   3:  SKNSH, enc_lr=5e-5, unfreeze=[4,5]
#   4:  HepG2, enc_lr=1e-4, unfreeze=[0,1,2,3,4,5] (all blocks)
#   5:  SKNSH, enc_lr=1e-4, unfreeze=[0,1,2,3,4,5]
#   6:  HepG2, enc_lr=5e-5, unfreeze=[0,1,2,3,4,5]
#   7:  SKNSH, enc_lr=5e-5, unfreeze=[0,1,2,3,4,5]
#   --- AG fold-1 ---
#   8:  HepG2, enc_lr=1e-4, unfreeze=[4,5]
#   9:  SKNSH, enc_lr=1e-4, unfreeze=[4,5]
#   10: HepG2, enc_lr=1e-4, unfreeze=[0,1,2,3,4,5]
#   11: SKNSH, enc_lr=1e-4, unfreeze=[0,1,2,3,4,5]
#   --- NTv3 ---
#   12: HepG2, enc_lr=5e-5
#   13: SKNSH, enc_lr=5e-5
#   14: HepG2, enc_lr=2e-4
#   15: SKNSH, enc_lr=2e-4
#
# Usage:
#   sbatch --array=0-15 scripts/slurm/s2_comprehensive_sweep.sh
#
#SBATCH --job-name=s2_full_sweep
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=default
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=200G

set -euo pipefail

set +u
source /etc/profile.d/modules.sh
set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
source scripts/slurm/setup_hpc_deps.sh

export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_command_buffer="

T=$SLURM_ARRAY_TASK_ID

echo "=== S2 Comprehensive Sweep task=$T ==="
echo "Node: $SLURMD_NODENAME  Date: $(date)"

# Setup data symlinks for all cells
for CELL in hepg2 sknsh; do
    mkdir -p "data/${CELL}"
    ln -sf "$(pwd)/data/k562/DATA-Table_S2__MPRA_dataset.txt" "data/${CELL}/DATA-Table_S2__MPRA_dataset.txt" 2>/dev/null || true
    ln -sf "$(pwd)/data/k562/hashfrag_splits" "data/${CELL}/hashfrag_splits" 2>/dev/null || true
    # Ensure test sets exist
    if [[ ! -f "data/${CELL}/test_sets/test_in_distribution_hashfrag.tsv" ]]; then
        uv run --no-sync python scripts/create_cellline_test_sets.py --cell-line "${CELL}"
    fi
done

# --- AG tasks (0-11): modify S2_CONFIG unfreeze_blocks via env var ---
# The exp1_1_scaling.py S2_CONFIG is hardcoded, so we need to pass
# unfreeze config differently. The cleanest way: use train_foundation_stage2.py
# for AG S2 since it supports config overrides.
# Actually, for AG we use exp1_1_scaling.py which reads S2_CONFIG.
# We'll modify the config by patching it at runtime.

if [ $T -le 11 ]; then
    # AG tasks
    CELLS=("hepg2" "sknsh" "hepg2" "sknsh" "hepg2" "sknsh" "hepg2" "sknsh" "hepg2" "sknsh" "hepg2" "sknsh")
    ENC_LRS=("1e-4" "1e-4" "5e-5" "5e-5" "1e-4" "1e-4" "5e-5" "5e-5" "1e-4" "1e-4" "1e-4" "1e-4")
    # Tasks 0-3: unfreeze=[4,5], Tasks 4-7: unfreeze=[0..5], Tasks 8-9: fold-1 [4,5], Tasks 10-11: fold-1 [0..5]
    UNFREEZE=("4,5" "4,5" "4,5" "4,5" "0,1,2,3,4,5" "0,1,2,3,4,5" "0,1,2,3,4,5" "0,1,2,3,4,5" "4,5" "4,5" "0,1,2,3,4,5" "0,1,2,3,4,5")
    # Tasks 0-7: all-folds, Tasks 8-11: fold-1
    IS_FOLD1=("0" "0" "0" "0" "0" "0" "0" "0" "1" "1" "1" "1")

    CELL="${CELLS[$T]}"
    ENC_LR="${ENC_LRS[$T]}"
    UF="${UNFREEZE[$T]}"
    FOLD1="${IS_FOLD1[$T]}"

    if [ "$FOLD1" == "1" ]; then
        export ALPHAGENOME_WEIGHTS="/grid/wsbs/home_norepl/christen/alphagenome_weights/alphagenome-jax-fold_1"
        MODEL_LABEL="fold_1"
    else
        MODEL_LABEL="all_folds"
    fi

    OUT_DIR="outputs/s2_sweep_v2/ag_${MODEL_LABEL}_${CELL}/elr${ENC_LR}_uf${UF}"
    echo "AG ${MODEL_LABEL} S2: cell=${CELL} enc_lr=${ENC_LR} unfreeze=${UF}"

    # Override S2_CONFIG unfreeze_blocks via environment variable
    export S2_UNFREEZE_BLOCKS="${UF}"

    uv run --no-sync python experiments/exp1_1_scaling.py \
        --task k562 \
        --student alphagenome_k562_s2 \
        --oracle ground_truth \
        --cell-line "${CELL}" \
        --reservoir genomic \
        --n-replicates 1 \
        --no-hp-sweep \
        --seed 42 \
        --output-dir "${OUT_DIR}" \
        --training-sizes 319742 \
        --epochs 50 \
        --early-stop-patience 10

elif [ $T -ge 12 ] && [ $T -le 15 ]; then
    # NTv3 tasks
    CELLS=("hepg2" "sknsh" "hepg2" "sknsh")
    ENC_LRS=("5e-5" "5e-5" "2e-4" "2e-4")
    IDX=$((T - 12))
    CELL="${CELLS[$IDX]}"
    ENC_LR="${ENC_LRS[$IDX]}"

    OUT_DIR="outputs/s2_sweep_v2/ntv3_post_${CELL}/elr${ENC_LR}"
    echo "NTv3 S2: cell=${CELL} enc_lr=${ENC_LR}"

    uv run --no-sync python experiments/train_foundation_stage2.py \
        model_name=ntv3_post \
        stage1_result_dir="outputs/ntv3_post_${CELL}_cached/seed_0/seed_0" \
        output_dir="${OUT_DIR}" \
        data_path="data/${CELL}" \
        cell_line="${CELL}" \
        seed=42 \
        epochs=50 \
        batch_size=4 \
        grad_accum_steps=2 \
        head_lr=0.001 \
        encoder_lr="${ENC_LR}" \
        weight_decay=1e-6 \
        hidden_dim=512 \
        dropout=0.1 \
        early_stop_patience=5 \
        max_train_sequences=20000 \
        max_val_sequences=2000 \
        rc_aug=True \
        unfreeze_mode=all \
        grad_clip=1.0 \
        amp_mode=bfloat16
fi

echo "Done: $(date)"
