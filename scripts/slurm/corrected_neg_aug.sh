#!/bin/bash
# Corrected neg-aug experiments with Gosai-scale labels derived from
# inactive-region-specific piecewise linear QQ transform.
#
# Tests 3 label scales × approaches that previously showed significant
# improvement in random/intergenic prediction:
#
#   0-2: Piecewise labels (mean=-0.16, std=0.36) with frac 2%, 5%, 10%
#   3-5: Global QQ labels (mean=-0.25, std=0.58) with frac 2%, 5%, 10%
#   6:   Piecewise labels, neg_plus_ood approach (OOD anchoring)
#   7:   Piecewise labels, zero_label approach (label=0 for all negatives)
#   8:   Piecewise, frac=5% + encoder_lr=1e-5 (lower encoder disruption)
#   9:   Global QQ, frac=5% + encoder_lr=1e-5
#  10:   Piecewise, frac=5% + weighted_loss 2x on negatives
#  11:   Piecewise, frac=10% dinuc-only (no random/gc, tighter composition)
#
# Submit:
#   /cm/shared/apps/slurm/current/bin/sbatch --array=0-11 scripts/slurm/corrected_neg_aug.sh
#
#SBATCH --job-name=corr_neg
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --time=02:00:00
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
NEG_BASE="data/synthetic_negatives_corrected"

# Common training args
COMMON="--config-name stage2_k562_oracle ++fold_id=0 ++n_folds=10 ++stage1_dir=${S1_DIR} ++use_full_dataset=True ++wandb_mode=offline"

case $T in
    # ══════════════════════════════════════════════════════════════════
    # Piecewise labels (mean=-0.16, std=0.36) at different fractions
    # ══════════════════════════════════════════════════════════════════
    0)
        LABEL="pw_frac02"
        OUT="outputs/oracle_neg_sweep/corrected/pw_frac02/fold_0"
        NEG_PATH="${NEG_BASE}/dinuc_shuffled_piecewise.tsv"
        FRAC=0.02
        ;;
    1)
        LABEL="pw_frac05"
        OUT="outputs/oracle_neg_sweep/corrected/pw_frac05/fold_0"
        NEG_PATH="${NEG_BASE}/dinuc_shuffled_piecewise.tsv"
        FRAC=0.05
        ;;
    2)
        LABEL="pw_frac10"
        OUT="outputs/oracle_neg_sweep/corrected/pw_frac10/fold_0"
        NEG_PATH="${NEG_BASE}/dinuc_shuffled_piecewise.tsv"
        FRAC=0.10
        ;;
    # ══════════════════════════════════════════════════════════════════
    # Global QQ labels (mean=-0.25, std=0.58) at different fractions
    # ══════════════════════════════════════════════════════════════════
    3)
        LABEL="qq_frac02"
        OUT="outputs/oracle_neg_sweep/corrected/qq_frac02/fold_0"
        NEG_PATH="${NEG_BASE}/dinuc_shuffled_global_qq.tsv"
        FRAC=0.02
        ;;
    4)
        LABEL="qq_frac05"
        OUT="outputs/oracle_neg_sweep/corrected/qq_frac05/fold_0"
        NEG_PATH="${NEG_BASE}/dinuc_shuffled_global_qq.tsv"
        FRAC=0.05
        ;;
    5)
        LABEL="qq_frac10"
        OUT="outputs/oracle_neg_sweep/corrected/qq_frac10/fold_0"
        NEG_PATH="${NEG_BASE}/dinuc_shuffled_global_qq.tsv"
        FRAC=0.10
        ;;
    # ══════════════════════════════════════════════════════════════════
    # Special approaches with piecewise labels
    # ══════════════════════════════════════════════════════════════════
    6)
        # neg_plus_ood: include OOD high-activity sequences as positive anchors
        # This was the Pareto-optimal config (rand=0.324, OOD=0.701)
        LABEL="pw_neg_plus_ood"
        OUT="outputs/oracle_neg_sweep/corrected/pw_neg_plus_ood/fold_0"
        NEG_PATH="${NEG_BASE}/negatives_piecewise.tsv"
        FRAC=0.05
        ;;
    7)
        # zero_label: all negatives get label=0.0
        # Previous zero_label: rand=0.251, OOD=0.702
        LABEL="pw_zero_label"
        OUT="outputs/oracle_neg_sweep/corrected/pw_zero_label/fold_0"
        NEG_PATH="${NEG_BASE}/dinuc_shuffled_zero_label.tsv"
        FRAC=0.05
        ;;
    8)
        # Lower encoder LR (1e-5 vs 1e-4) to reduce encoder disruption
        LABEL="pw_frac05_elr5"
        OUT="outputs/oracle_neg_sweep/corrected/pw_frac05_elr5/fold_0"
        NEG_PATH="${NEG_BASE}/dinuc_shuffled_piecewise.tsv"
        FRAC=0.05
        ;;
    9)
        # Global QQ + lower encoder LR
        LABEL="qq_frac05_elr5"
        OUT="outputs/oracle_neg_sweep/corrected/qq_frac05_elr5/fold_0"
        NEG_PATH="${NEG_BASE}/dinuc_shuffled_global_qq.tsv"
        FRAC=0.05
        ;;
    10)
        # Piecewise, all negatives combined (random + dinuc + gc), frac=5%
        LABEL="pw_frac05_all"
        OUT="outputs/oracle_neg_sweep/corrected/pw_frac05_all/fold_0"
        NEG_PATH="${NEG_BASE}/negatives_piecewise.tsv"
        FRAC=0.05
        ;;
    11)
        # Piecewise, 10% dinuc-only (tighter composition match)
        LABEL="pw_frac10_dinuc_only"
        OUT="outputs/oracle_neg_sweep/corrected/pw_frac10_dinuc_only/fold_0"
        NEG_PATH="${NEG_BASE}/dinuc_shuffled_piecewise.tsv"
        FRAC=0.10
        ;;
esac

# Skip if done
[ -f "${OUT}/test_metrics.json" ] && echo "SKIP: ${LABEL}" && exit 0

echo "=== ${LABEL} — $(date) ==="

# Build command
CMD="uv run --no-sync python experiments/train_stage2_k562_hashfrag.py ${COMMON}"
CMD="${CMD} ++output_dir=${OUT}"
CMD="${CMD} ++negatives_path=${NEG_PATH}"
CMD="${CMD} ++neg_fraction=${FRAC}"

# Special config overrides
case $T in
    8|9)
        # Lower encoder LR
        CMD="${CMD} ++encoder_lr=1e-5"
        ;;
esac

echo "CMD: ${CMD}"
eval ${CMD}

echo "=== ${LABEL} DONE — $(date) ==="
