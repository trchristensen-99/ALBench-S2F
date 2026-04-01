#!/bin/bash
# Augmentation sweep: test RC-only vs RC+Shift vs Shift-only for each model.
# Uses chr_split on K562 (single cell to reduce cost, 3 seeds each).
#
# Following Alan's alphagenome_FT_MPRA approach:
#   - RC: 50% probability reverse complement
#   - Shift: 50% probability random shift ±15bp
#   - Both applied during training, not eval
#
# Configs per model:
#   A = baseline (RC only, current default)
#   B = RC + Shift(±15bp)
#   C = Shift only (no RC)
#   D = No augmentation
#
# Array:
#   0-3:   DREAM-RNN K562 (A, B, C, D)
#   4-7:   Malinois K562 (A, B, C, D)
#   8-11:  Enformer S2 K562 (A, B, C, D)
#
# Submit across QoS tiers:
#   sbatch --array=0-3 --qos=fast scripts/slurm/augmentation_sweep.sh
#   sbatch --array=4-7 --qos=default scripts/slurm/augmentation_sweep.sh
#   sbatch --array=8-11 --qos=slow_nice scripts/slurm/augmentation_sweep.sh
#
#SBATCH --job-name=aug_sweep
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

T=$SLURM_ARRAY_TASK_ID
echo "=== aug_sweep task=${T} node=${SLURMD_NODENAME} date=$(date) ==="

# Config index within model group
CFG_IDX=$((T % 4))
MODEL_GROUP=$((T / 4))

# Augmentation configs: (rc_flag, shift_flag, label)
case ${CFG_IDX} in
    0) RC="true";  SHIFT="false"; LABEL="rc_only" ;;
    1) RC="true";  SHIFT="true";  LABEL="rc_shift" ;;
    2) RC="false"; SHIFT="true";  LABEL="shift_only" ;;
    3) RC="false"; SHIFT="false"; LABEL="no_aug" ;;
esac

echo "Config: ${LABEL} (RC=${RC}, Shift=${SHIFT})"

# ── DREAM-RNN ──
if [ ${MODEL_GROUP} -eq 0 ]; then
    echo "Model: DREAM-RNN K562, Aug: ${LABEL}"
    OUT="outputs/aug_sweep/dream_rnn/${LABEL}"

    # DREAM-RNN RC is controlled by the model (always bidirectional).
    # For "no RC", we'd need to modify the code. For now, RC is always on
    # for DREAM-RNN (it's built into the architecture), so we only sweep shift.
    SHIFT_FLAG=""
    [[ "${SHIFT}" == "true" ]] && SHIFT_FLAG="--shift-aug --max-shift 15"

    uv run --no-sync python experiments/exp1_1_scaling.py \
        --task k562 --student dream_rnn \
        --oracle ground_truth --reservoir genomic --chr-split \
        --n-replicates 3 --seed 42 \
        --output-dir "${OUT}" \
        --training-sizes 400000 --epochs 80 --ensemble-size 1 \
        --early-stop-patience 10 \
        ${SHIFT_FLAG}

# ── Malinois ──
elif [ ${MODEL_GROUP} -eq 1 ]; then
    echo "Model: Malinois K562, Aug: ${LABEL}"
    OUT="outputs/aug_sweep/malinois/${LABEL}"

    for SEED in 0 1 2; do
        echo "--- seed ${SEED} ---"
        uv run --no-sync python experiments/train_malinois_k562.py \
            ++data_path="data/k562" \
            ++output_dir="${OUT}/seed_${SEED}" \
            ++seed="${SEED}" \
            ++cell_line="k562" \
            ++chr_split=True \
            ++use_reverse_complement="${RC}" \
            ++shift_aug="${SHIFT}" \
            ++max_shift=15
    done

# ── Enformer S2 ──
elif [ ${MODEL_GROUP} -eq 2 ]; then
    echo "Model: Enformer S2 K562, Aug: ${LABEL}"
    OUT="outputs/aug_sweep/enformer_s2/${LABEL}"

    # Use existing S1 checkpoint as warm start
    S1_DIR="outputs/chr_split/k562/enformer_s1/seed_42"
    if [ ! -d "${S1_DIR}" ]; then
        S1_DIR="outputs/chr_split/k562/enformer_s1_v2/seed_42"
    fi

    for SEED in 42 123 456; do
        echo "--- seed ${SEED} ---"
        uv run --no-sync python experiments/train_foundation_stage2.py \
            ++model_name=enformer \
            ++stage1_result_dir="${S1_DIR}" \
            ++output_dir="${OUT}/seed_${SEED}" \
            ++data_path="data/k562" \
            ++cell_line="k562" \
            ++chr_split=True \
            ++seed="${SEED}" \
            ++epochs=15 \
            ++batch_size=4 \
            ++grad_accum_steps=2 \
            ++head_lr=0.001 \
            ++encoder_lr=0.0001 \
            ++weight_decay=1e-6 \
            ++hidden_dim=512 \
            ++dropout=0.1 \
            ++early_stop_patience=5 \
            ++max_train_sequences=20000 \
            ++max_val_sequences=2000 \
            ++rc_aug="${RC}" \
            ++shift_aug="${SHIFT}" \
            ++max_shift=15 \
            ++unfreeze_mode=all \
            ++grad_clip=1.0 \
            ++amp_mode=bfloat16
    done
fi

echo "=== task=${T} DONE — $(date) ==="
