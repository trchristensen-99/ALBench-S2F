#!/bin/bash
# Principled CpG debiasing experiments.
#
# Multiple approaches to break the spurious CpG→activity confound:
#
#   0: cpg_depleted_aug_5pct  — Add CpG-depleted versions of training seqs (same labels)
#                                Teaches: removing CpG doesn't change activity
#   1: cpg_enriched_aug_5pct  — Add CpG-enriched versions of training seqs (same labels)
#                                Teaches: adding CpG doesn't change activity
#   2: cpg_both_aug_5pct      — Both depleted + enriched (same labels)
#                                Strongest: CpG is irrelevant to activity
#   3: cpg_both_aug_10pct     — Same as 2 but 10% fraction
#   4: cpg_both_aug_2pct      — Same as 2 but 2% fraction
#   5: high_cpg_ctrl_neg_5pct — High-CpG random seqs + Gosai ctrl_neg labels (+0.27)
#                                Combines CpG targeting with correct labels
#   6: cpg_both_plus_neg_5pct — cpg_both_aug + high-CpG random negatives (combined approach)
#
# Submit:
#   /cm/shared/apps/slurm/current/bin/sbatch --array=0-6 scripts/slurm/principled_cpg_fix.sh
#
#SBATCH --job-name=cpg_fix
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

COMMON="--config-name stage2_k562_oracle ++fold_id=0 ++n_folds=10 ++stage1_dir=${S1_DIR} ++use_full_dataset=True ++wandb_mode=offline"

case $T in
    0)
        LABEL="cpg_depleted_aug"
        OUT="outputs/oracle_neg_sweep/principled_cpg/cpg_depleted_aug_frac05/fold_0"
        NEG_PATH="data/cpg_augmentation/cpg_depleted_aug.tsv"
        FRAC=0.05
        ;;
    1)
        LABEL="cpg_enriched_aug"
        OUT="outputs/oracle_neg_sweep/principled_cpg/cpg_enriched_aug_frac05/fold_0"
        NEG_PATH="data/cpg_augmentation/cpg_enriched_aug.tsv"
        FRAC=0.05
        ;;
    2)
        LABEL="cpg_both_aug"
        OUT="outputs/oracle_neg_sweep/principled_cpg/cpg_both_aug_frac05/fold_0"
        NEG_PATH="data/cpg_augmentation/cpg_both_aug.tsv"
        FRAC=0.05
        ;;
    3)
        LABEL="cpg_both_aug_10pct"
        OUT="outputs/oracle_neg_sweep/principled_cpg/cpg_both_aug_frac10/fold_0"
        NEG_PATH="data/cpg_augmentation/cpg_both_aug.tsv"
        FRAC=0.10
        ;;
    4)
        LABEL="cpg_both_aug_2pct"
        OUT="outputs/oracle_neg_sweep/principled_cpg/cpg_both_aug_frac02/fold_0"
        NEG_PATH="data/cpg_augmentation/cpg_both_aug.tsv"
        FRAC=0.02
        ;;
    5)
        LABEL="high_cpg_ctrl_neg"
        OUT="outputs/oracle_neg_sweep/principled_cpg/high_cpg_ctrl_neg_frac05/fold_0"
        NEG_PATH="data/synthetic_negatives_cpg_aware/high_cpg_only.tsv"
        FRAC=0.05
        ;;
    6)
        # Combined: cpg_both_aug (teaches CpG-invariance on real seqs)
        # + high_cpg random negatives (teaches high-CpG random = inactive)
        # We'll concatenate the two files on the fly
        LABEL="cpg_both_plus_neg"
        OUT="outputs/oracle_neg_sweep/principled_cpg/cpg_both_plus_neg_frac05/fold_0"
        # Create combined file
        COMBINED="/tmp/cpg_combined_$$$.tsv"
        head -1 data/cpg_augmentation/cpg_both_aug.tsv > "$COMBINED"
        tail -n +2 data/cpg_augmentation/cpg_both_aug.tsv >> "$COMBINED"
        tail -n +2 data/synthetic_negatives_cpg_aware/high_cpg_only.tsv >> "$COMBINED"
        NEG_PATH="$COMBINED"
        FRAC=0.05
        ;;
esac

[ -f "${OUT}/test_metrics.json" ] && echo "SKIP: ${LABEL}" && exit 0

echo "=== ${LABEL} — $(date) ==="

uv run --no-sync python experiments/train_stage2_k562_hashfrag.py \
    ${COMMON} \
    ++output_dir="${OUT}" \
    ++negatives_path="${NEG_PATH}" \
    ++neg_fraction="${FRAC}"

echo "=== ${LABEL} DONE — $(date) ==="
