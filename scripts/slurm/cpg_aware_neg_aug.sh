#!/bin/bash
# CpG-aware neg-aug experiments.
#
# The oracle has a spurious CpG->activity slope of 11.4 log2FC/unit,
# while real controls show ~0. These experiments directly target
# this confound by providing negatives at various CpG levels
# with matching Gosai-native labels.
#
# 5 strategies × 2 fractions = 10 experiments:
#   0-1: cpg_uniform       (CpG-varied random, label=+0.20) at 2%, 5%
#   2-3: cpg_ctrl_neg      (CpG-varied random, label=+0.27) at 2%, 5%
#   4-5: high_cpg_only     (CpG 0.04-0.08 only, label=+0.20) at 2%, 5%
#   6-7: cpg_mixed         (depleted + natural 50/50, label=+0.25) at 2%, 5%
#   8-9: dinuc_ctrl_neg_label (dinuc-shuffled genomic, label=+0.27) at 2%, 5%
#
# Submit:
#   /cm/shared/apps/slurm/current/bin/sbatch --array=0-9 scripts/slurm/cpg_aware_neg_aug.sh
#
#SBATCH --job-name=cpg_neg
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
NEG_BASE="data/synthetic_negatives_cpg_aware"

COMMON="--config-name stage2_k562_oracle ++fold_id=0 ++n_folds=10 ++stage1_dir=${S1_DIR} ++use_full_dataset=True ++wandb_mode=offline"

NAMES=("cpg_uniform" "cpg_uniform" "cpg_ctrl_neg" "cpg_ctrl_neg" "high_cpg_only" "high_cpg_only" "cpg_mixed" "cpg_mixed" "dinuc_ctrl_neg_label" "dinuc_ctrl_neg_label")
FRACS=("0.02" "0.05" "0.02" "0.05" "0.02" "0.05" "0.02" "0.05" "0.02" "0.05")

NAME=${NAMES[$T]}
FRAC=${FRACS[$T]}
FRAC_PCT=$(echo "$FRAC" | sed 's/0\.//')

OUT="outputs/oracle_neg_sweep/cpg_aware/${NAME}_frac${FRAC_PCT}/fold_0"
NEG_PATH="${NEG_BASE}/${NAME}.tsv"

[ -f "${OUT}/test_metrics.json" ] && echo "SKIP: ${NAME}_frac${FRAC_PCT}" && exit 0

echo "=== ${NAME}_frac${FRAC_PCT} — $(date) ==="

uv run --no-sync python experiments/train_stage2_k562_hashfrag.py \
    ${COMMON} \
    ++output_dir="${OUT}" \
    ++negatives_path="${NEG_PATH}" \
    ++neg_fraction="${FRAC}"

echo "=== ${NAME}_frac${FRAC_PCT} DONE — $(date) ==="
