#!/bin/bash
# Combined debiasing: loss-based + neg-aug approaches.
#
# Best individual results:
#   - spectral λ=0.5: OOD=0.767, rand=+0.477 (best loss-based)
#   - negaug_random_only 5%: OOD=0.508, rand=+0.316 (best bias reduction)
#   - counterfactual λ=0.3: OOD=0.782, rand=+0.701 (best OOD)
#
# Combined configs (10 total):
#   0: spectral_05 + random_only 2%
#   1: spectral_05 + random_only 5%
#   2: spectral_05 + motif_full 2%
#   3: spectral_05 + motif_full 5%
#   4: counterfactual_03 + random_only 2%
#   5: counterfactual_03 + random_only 5%
#   6: cpg_invariance_05 + random_only 2%
#   7: spectral_03 + random_only 3%  (moderate both)
#   8: spectral_10 + random_only 1%  (light both)
#   9: spectral_05 + random_only 2% + longer training (25ep)
#
#SBATCH --job-name=db_comb
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

NEG_DIR="data/neg_aug_motif_conditional"

declare -a NAMES=(
    "combo_spectral05_random2pct"
    "combo_spectral05_random5pct"
    "combo_spectral05_motif2pct"
    "combo_spectral05_motif5pct"
    "combo_counterfactual03_random2pct"
    "combo_counterfactual03_random5pct"
    "combo_cpginv05_random2pct"
    "combo_spectral03_random3pct"
    "combo_spectral10_random1pct"
    "combo_spectral05_random2pct_25ep"
)

declare -a DEBIAS_MODES=("spectral" "spectral" "spectral" "spectral" "counterfactual_consistency" "counterfactual_consistency" "cpg_invariance" "spectral" "spectral" "spectral")
declare -a LAMBDAS=(0.5 0.5 0.5 0.5 0.3 0.3 0.5 0.3 1.0 0.5)
declare -a NEG_PATHS=(
    "${NEG_DIR}/random_inactive_only.tsv"
    "${NEG_DIR}/random_inactive_only.tsv"
    "${NEG_DIR}/motif_conditional_negaug.tsv"
    "${NEG_DIR}/motif_conditional_negaug.tsv"
    "${NEG_DIR}/random_inactive_only.tsv"
    "${NEG_DIR}/random_inactive_only.tsv"
    "${NEG_DIR}/random_inactive_only.tsv"
    "${NEG_DIR}/random_inactive_only.tsv"
    "${NEG_DIR}/random_inactive_only.tsv"
    "${NEG_DIR}/random_inactive_only.tsv"
)
declare -a NEG_FRACS=(0.02 0.05 0.02 0.05 0.02 0.05 0.02 0.03 0.01 0.02)
declare -a EPOCHS=(15 15 15 15 15 15 15 15 15 25)

NAME=${NAMES[$T]}
DEBIAS=${DEBIAS_MODES[$T]}
LAMBDA=${LAMBDAS[$T]}
NEG_PATH=${NEG_PATHS[$T]}
NEG_FRAC=${NEG_FRACS[$T]}
EP=${EPOCHS[$T]}

OUT="outputs/debias_sweep/${NAME}"
[ -f "${OUT}/test_metrics.json" ] && echo "SKIP: ${NAME} already done" && exit 0

echo "=== Combined Debias: ${NAME} — $(date) ==="

uv run --no-sync python experiments/train_stage2_k562_hashfrag.py \
    --config-name stage2_k562_full_train \
    variant=s2c \
    encoder_lr=1e-4 \
    head_lr=1e-3 \
    epochs=${EP} \
    ++batch_size=128 \
    output_dir="${OUT}" \
    ++fold_id=0 \
    ++debias_mode="${DEBIAS}" \
    ++debias_lambda=${LAMBDA} \
    ++negatives_path="${NEG_PATH}" \
    ++neg_fraction=${NEG_FRAC}

echo "=== DONE — $(date) ==="
