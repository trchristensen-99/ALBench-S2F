#!/bin/bash
# Comprehensive debiasing sweep: test all approaches on fast/default queues.
#
# Approaches (12 total):
#   Loss-based debiasing (existing + new):
#     0: spectral (lambda=0.1)
#     1: spectral (lambda=0.5)
#     2: group_dro (lambda=0.3)
#     3: cpg_invariance (lambda=0.5)
#     4: cpg_gradient_penalty (lambda=0.5)
#     5: counterfactual_consistency (lambda=0.3)
#     6: adaptive_group_dro (lambda=0.3)
#     7: conditional_invariance (lambda=0.5)
#
#   Neg-aug data approaches:
#     8:  motif_conditional full (5%)
#     9:  motif_conditional full (10%)
#     10: random_inactive_only (5%)
#     11: motif_plus_random (5%)
#
# Each trains 1 fold (~2-3h on H100), evaluates on test + bias metrics.
#
#SBATCH --job-name=debias
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=fast
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
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

# Generate neg-aug data if it doesn't exist
NEG_DIR="data/neg_aug_motif_conditional"
if [ ! -f "${NEG_DIR}/motif_conditional_negaug.tsv" ]; then
    echo "Generating motif-conditional neg-aug data..."
    uv run --no-sync python scripts/generate_motif_conditional_negaug.py --output-dir "${NEG_DIR}"
fi

# Define all configurations
declare -a NAMES=(
    "spectral_l01"
    "spectral_l05"
    "group_dro_l03"
    "cpg_invariance_l05"
    "cpg_gradient_penalty_l05"
    "counterfactual_l03"
    "adaptive_group_dro_l03"
    "conditional_invariance_l05"
    "negaug_motif_full_5pct"
    "negaug_motif_full_10pct"
    "negaug_random_only_5pct"
    "negaug_motif_plus_random_5pct"
)

declare -a DEBIAS_MODES=(
    "spectral"
    "spectral"
    "group_dro"
    "cpg_invariance"
    "cpg_gradient_penalty"
    "counterfactual_consistency"
    "adaptive_group_dro"
    "conditional_invariance"
    "none"
    "none"
    "none"
    "none"
)

declare -a LAMBDAS=(0.1 0.5 0.3 0.5 0.5 0.3 0.3 0.5 0 0 0 0)

declare -a NEG_PATHS=(
    "none" "none" "none" "none" "none" "none" "none" "none"
    "${NEG_DIR}/motif_conditional_negaug.tsv"
    "${NEG_DIR}/motif_conditional_negaug.tsv"
    "${NEG_DIR}/random_inactive_only.tsv"
    "${NEG_DIR}/motif_plus_random.tsv"
)

declare -a NEG_FRACS=(0 0 0 0 0 0 0 0 0.05 0.10 0.05 0.05)

NAME=${NAMES[$T]}
DEBIAS=${DEBIAS_MODES[$T]}
LAMBDA=${LAMBDAS[$T]}
NEG_PATH=${NEG_PATHS[$T]}
NEG_FRAC=${NEG_FRACS[$T]}

OUT="outputs/debias_sweep/${NAME}/fold_0"
[ -f "${OUT}/test_metrics.json" ] && echo "SKIP: ${NAME} already done" && exit 0

echo "=== Debias: ${NAME} (mode=${DEBIAS} lambda=${LAMBDA}) — $(date) ==="

# Build command
CMD="uv run --no-sync python experiments/train_stage2_k562_hashfrag.py"
CMD+=" variant=s2c"
CMD+=" encoder_lr=1e-4"
CMD+=" head_lr=1e-3"
CMD+=" epochs=15"
CMD+=" batch_size=128"
CMD+=" use_dedicated_val=true"
CMD+=" output_dir=outputs/debias_sweep/${NAME}"
CMD+=" fold_id=0"
CMD+=" debias_mode=${DEBIAS}"
CMD+=" debias_lambda=${LAMBDA}"

if [ "${NEG_PATH}" != "none" ]; then
    CMD+=" negatives_path=${NEG_PATH}"
    CMD+=" neg_fraction=${NEG_FRAC}"
fi

eval $CMD

echo "=== DONE — $(date) ==="
