#!/bin/bash
# Breakthrough debiasing: push past the +0.4 random DNA barrier.
#
# Tests all available levers simultaneously:
# - Higher augmentation fractions (30-50%)
# - Longer training (100 epochs, patience 30)
# - Higher encoder LR (5e-4, 1e-3)
# - More unfrozen encoder blocks (all 6 vs just top 2)
# - Combined approaches
#
# Uses the 500K breakthrough dataset (enriched + depleted + multi-CpG negatives)
#
#  0: baseline_reference (no aug, standard config)
#  1: aug_40pct (standard LR, standard blocks)
#  2: aug_50pct
#  3: aug_40pct + longer (epochs=100, patience=30)
#  4: aug_40pct + higher_enc_lr (5e-4)
#  5: aug_40pct + much_higher_enc_lr (1e-3)
#  6: aug_40pct + all_blocks (unfreeze all encoder blocks)
#  7: aug_40pct + all_blocks + higher_enc_lr
#  8: aug_50pct + longer + higher_enc_lr
#  9: aug_50pct + all_blocks + longer + higher_enc_lr (FULL KITCHEN SINK)
# 10: aug_40pct + cpg_invariance + longer
# 11: aug_30pct + all_blocks + longer (moderate)
# 12: aug_40pct + higher_head_lr (5e-3) — bigger head updates
# 13: aug_40pct + lower_enc_lr (1e-5) + longer — slow encoder, long training
# 14: aug_50pct + higher_enc_lr + spectral
#
# Submit:
#   /cm/shared/apps/slurm/current/bin/sbatch --array=0-14 scripts/slurm/breakthrough_debiasing.sh
#
#SBATCH --job-name=brkthru
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --time=06:00:00
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
export ALPHAGENOME_WEIGHTS="/grid/wsbs/home_norepl/christen/alphagenome_weights/alphagenome-jax-all_folds-v1"

T=$SLURM_ARRAY_TASK_ID
S1_DIR="outputs/oracle_full_856k/s1/oracle_0"
NEG_PATH="data/neg_aug_breakthrough/breakthrough_500k.tsv"

# Base config
EPOCHS=50
PATIENCE=10
ENC_LR=0.0001
HEAD_LR=0.001
BLOCKS="++unfreeze_encoder_blocks=[4,5]"
FRAC=0.40
DEBIAS=""
NEG_ARGS="++negatives_path=${NEG_PATH} ++neg_fraction=${FRAC}"

case $T in
    0)  # Baseline reference — no augmentation
        LABEL="baseline_ref"
        NEG_ARGS=""
        ;;
    1)  # Standard config, 40% aug
        LABEL="aug40"
        ;;
    2)  # Standard config, 50% aug
        LABEL="aug50"
        FRAC=0.50
        NEG_ARGS="++negatives_path=${NEG_PATH} ++neg_fraction=${FRAC}"
        ;;
    3)  # 40% aug + longer training
        LABEL="aug40_long"
        EPOCHS=100
        PATIENCE=30
        ;;
    4)  # 40% aug + higher encoder LR
        LABEL="aug40_elr5e4"
        ENC_LR=0.0005
        ;;
    5)  # 40% aug + much higher encoder LR
        LABEL="aug40_elr1e3"
        ENC_LR=0.001
        ;;
    6)  # 40% aug + unfreeze all encoder blocks
        LABEL="aug40_allblocks"
        BLOCKS=""  # empty = unfreeze all
        ;;
    7)  # 40% aug + all blocks + higher encoder LR
        LABEL="aug40_allblocks_elr5e4"
        BLOCKS=""
        ENC_LR=0.0005
        ;;
    8)  # 50% aug + longer + higher encoder LR
        LABEL="aug50_long_elr5e4"
        FRAC=0.50
        NEG_ARGS="++negatives_path=${NEG_PATH} ++neg_fraction=${FRAC}"
        EPOCHS=100
        PATIENCE=30
        ENC_LR=0.0005
        ;;
    9)  # FULL KITCHEN SINK: 50% + all blocks + longer + higher LR
        LABEL="kitchen_sink_full"
        FRAC=0.50
        NEG_ARGS="++negatives_path=${NEG_PATH} ++neg_fraction=${FRAC}"
        EPOCHS=100
        PATIENCE=30
        ENC_LR=0.0005
        BLOCKS=""
        ;;
    10) # 40% aug + CpG invariance loss + longer
        LABEL="aug40_cpginv_long"
        EPOCHS=100
        PATIENCE=30
        DEBIAS="++debias_mode=cpg_invariance ++debias_lambda=1.0"
        ;;
    11) # 30% aug + all blocks + longer (moderate)
        LABEL="aug30_allblocks_long"
        FRAC=0.30
        NEG_ARGS="++negatives_path=${NEG_PATH} ++neg_fraction=${FRAC}"
        EPOCHS=100
        PATIENCE=30
        BLOCKS=""
        ;;
    12) # 40% aug + higher head LR
        LABEL="aug40_hlr5e3"
        HEAD_LR=0.005
        ;;
    13) # 40% aug + very low encoder LR + very long training
        LABEL="aug40_elr1e5_vlong"
        ENC_LR=0.00001
        EPOCHS=150
        PATIENCE=50
        ;;
    14) # 50% aug + higher LR + spectral
        LABEL="aug50_elr5e4_spectral"
        FRAC=0.50
        NEG_ARGS="++negatives_path=${NEG_PATH} ++neg_fraction=${FRAC}"
        ENC_LR=0.0005
        DEBIAS="++debias_mode=spectral ++debias_lambda=0.05"
        ;;
esac

OUT="outputs/oracle_neg_sweep/breakthrough/${LABEL}/fold_0"
[ -f "${OUT}/test_metrics.json" ] && echo "SKIP: ${LABEL}" && exit 0

echo "=== ${LABEL} — $(date) ==="
echo "  enc_lr=${ENC_LR} head_lr=${HEAD_LR} epochs=${EPOCHS} patience=${PATIENCE}"
echo "  frac=${FRAC} blocks=${BLOCKS:-all}"

CMD="uv run --no-sync python experiments/train_stage2_k562_hashfrag.py"
CMD="${CMD} --config-name stage2_k562_oracle"
CMD="${CMD} ++fold_id=0 ++n_folds=10"
CMD="${CMD} ++stage1_dir=${S1_DIR}"
CMD="${CMD} ++output_dir=${OUT}"
CMD="${CMD} ++use_full_dataset=True ++wandb_mode=offline"
CMD="${CMD} ++encoder_lr=${ENC_LR} ++head_lr=${HEAD_LR}"
CMD="${CMD} ++epochs=${EPOCHS} ++early_stop_patience=${PATIENCE}"
if [ -n "${BLOCKS}" ]; then
    CMD="${CMD} ${BLOCKS}"
fi
if [ -n "${NEG_ARGS}" ]; then
    CMD="${CMD} ${NEG_ARGS}"
fi
if [ -n "${DEBIAS}" ]; then
    CMD="${CMD} ${DEBIAS}"
fi

echo "CMD: ${CMD}"
eval ${CMD}

echo "=== ${LABEL} DONE — $(date) ==="
