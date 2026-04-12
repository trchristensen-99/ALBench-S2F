#!/bin/bash
# Neg-aug experiments using Gosai-native ctrl_neg baseline.
#
# Key finding: Gosai episomal negative controls have mean=+0.27
# (not -0.16 or -0.45 as derived from Agarwal lentiMPRA).
# CpG content of random DNA (0.0625) predicts even higher activity
# on unmethylated episomes.
#
# Test 4 label strategies × 2 fractions = 8 experiments:
#   0-1: ctrl_neg (mean=+0.27, std=0.49) at 2%, 5%
#   2-3: ctrl_neg_tight (mean=+0.19, std=0.30) at 2%, 5%
#   4-5: zero (mean=0.0, std=0.30) at 2%, 5%
#   6-7: intermediate (mean=+0.40, std=0.45) at 2%, 5%
#
# Submit:
#   /cm/shared/apps/slurm/current/bin/sbatch --array=0-7 scripts/slurm/gosai_native_neg_aug.sh
#
#SBATCH --job-name=gn_neg
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
NEG_BASE="data/synthetic_negatives_gosai_native"

COMMON="--config-name stage2_k562_oracle ++fold_id=0 ++n_folds=10 ++stage1_dir=${S1_DIR} ++use_full_dataset=True ++wandb_mode=offline"

# Label strategy and fraction
LABELS=("ctrl_neg" "ctrl_neg" "ctrl_neg_tight" "ctrl_neg_tight" "zero" "zero" "intermediate" "intermediate")
FRACS=("0.02" "0.05" "0.02" "0.05" "0.02" "0.05" "0.02" "0.05")

LABEL=${LABELS[$T]}
FRAC=${FRACS[$T]}
FRAC_PCT=$(echo "$FRAC" | sed 's/0\.//')

OUT="outputs/oracle_neg_sweep/gosai_native/${LABEL}_frac${FRAC_PCT}/fold_0"
NEG_PATH="${NEG_BASE}/dinuc_${LABEL}.tsv"

# Skip if done
[ -f "${OUT}/test_metrics.json" ] && echo "SKIP: ${LABEL}_frac${FRAC_PCT}" && exit 0

echo "=== ${LABEL}_frac${FRAC_PCT} — $(date) ==="

uv run --no-sync python experiments/train_stage2_k562_hashfrag.py \
    ${COMMON} \
    ++output_dir="${OUT}" \
    ++negatives_path="${NEG_PATH}" \
    ++neg_fraction="${FRAC}"

echo "=== ${LABEL}_frac${FRAC_PCT} DONE — $(date) ==="
