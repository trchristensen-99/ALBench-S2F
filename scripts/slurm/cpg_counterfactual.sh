#!/bin/bash
# CpG counterfactual augmentation experiments.
#
# Creates minimal CpG perturbations of real training sequences:
# CG↔GC swaps preserve nucleotide composition while changing CpG content.
# Same label for original and swapped versions teaches: CpG content alone
# doesn't determine activity — the regulatory CONTEXT matters.
#
# This is the cleanest counterfactual: identical GC%, identical nucleotide
# counts, only dinucleotide order changes. If the model learns from this,
# it will specifically unlearn "CpG frequency = activity" while preserving
# "CpG at a functional promoter = activity."
#
#   0: cpg_to_gpc (CG→GC, removes CpG) at 5%
#   1: gpc_to_cpg (GC→CG, adds CpG) at 5%
#   2: bidirectional (both) at 5%
#   3: bidirectional at 10%
#   4: bidirectional at 2%
#   5: bidirectional at 15%
#   6: bidirectional at 20%
#
# Submit:
#   /cm/shared/apps/slurm/current/bin/sbatch --array=0-6 scripts/slurm/cpg_counterfactual.sh
#
#SBATCH --job-name=cpg_cf
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

NAMES=("cpg_to_gpc" "gpc_to_cpg" "cpg_bidirectional" "cpg_bidirectional" "cpg_bidirectional" "cpg_bidirectional" "cpg_bidirectional")
FILES=("cpg_to_gpc_swap" "gpc_to_cpg_swap" "cpg_bidirectional_swap" "cpg_bidirectional_swap" "cpg_bidirectional_swap" "cpg_bidirectional_swap" "cpg_bidirectional_swap")
FRACS=("0.05" "0.05" "0.05" "0.10" "0.02" "0.15" "0.20")

NAME=${NAMES[$T]}
FILE=${FILES[$T]}
FRAC=${FRACS[$T]}
FRAC_PCT=$(echo "$FRAC" | sed 's/0\.//')

OUT="outputs/oracle_neg_sweep/counterfactual_cpg/${NAME}_frac${FRAC_PCT}/fold_0"
NEG_PATH="data/cpg_counterfactual/${FILE}.tsv"

[ -f "${OUT}/test_metrics.json" ] && echo "SKIP: ${NAME}_frac${FRAC_PCT}" && exit 0

echo "=== ${NAME}_frac${FRAC_PCT} — $(date) ==="

uv run --no-sync python experiments/train_stage2_k562_hashfrag.py \
    ${COMMON} \
    ++output_dir="${OUT}" \
    ++negatives_path="${NEG_PATH}" \
    ++neg_fraction="${FRAC}"

echo "=== ${NAME}_frac${FRAC_PCT} DONE — $(date) ==="
