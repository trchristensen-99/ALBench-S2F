#!/bin/bash
# Aggressive CpG debiasing — pushing harder on the best approaches.
#
# Based on finding that cpg_both_plus_neg 5% (rand=+0.48, OOD=0.749)
# is the Pareto-optimal approach. Now testing:
# - Higher fractions (10-30%) of enrichment augmentation
# - Larger CpG perturbations (+20-25 CpGs per sequence)
# - Multi-CpG-level random negatives (anchoring high-CpG region)
# - Combined approaches ("kitchen sink")
# - Loss modifications combined with the best data approaches
#
# Accept some OOD cost to push random DNA closer to ground truth.
#
#   0: best_combo_scaled 10%  (scaled-up best approach)
#   1: best_combo_scaled 15%
#   2: best_combo_scaled 20%
#   3: very_aggressive_enrich 10%  (+20 CpGs per seq)
#   4: very_aggressive_enrich 15%
#   5: both_large 15%  (bidirectional at high fraction)
#   6: both_large 25%
#   7: multi_cpg_negatives 5%  (anchoring negatives only)
#   8: multi_cpg_negatives 10%
#   9: kitchen_sink 10%  (enrichment + multi-CpG negatives)
#  10: kitchen_sink 15%
#  11: kitchen_sink 20%
#  12: enrich_inactive_only 10%  (only enrich inactive seqs)
#  13: enrich_inactive_only 15%
#  14: extreme_enrich 10%  (target CpG ~0.10)
#  15: extreme_enrich 15%
#  16: kitchen_sink 10% + spectral λ=0.05
#  17: kitchen_sink 10% + cpg_invariance λ=1.0
#  18: best_combo_scaled 15% + cpg_invariance λ=0.5
#  19: both_large 20% + spectral λ=0.05
#
# Submit:
#   /cm/shared/apps/slurm/current/bin/sbatch --array=0-19 scripts/slurm/aggressive_debiasing.sh
#
#SBATCH --job-name=agg_deb
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --time=02:30:00
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
NEG_BASE="data/neg_aug_aggressive"

COMMON="--config-name stage2_k562_oracle ++fold_id=0 ++n_folds=10 ++stage1_dir=${S1_DIR} ++use_full_dataset=True ++wandb_mode=offline"
DEBIAS_ARGS=""

case $T in
    0)  NAME="best_combo_scaled"; FRAC=0.10 ;;
    1)  NAME="best_combo_scaled"; FRAC=0.15 ;;
    2)  NAME="best_combo_scaled"; FRAC=0.20 ;;
    3)  NAME="very_aggressive_enrich"; FRAC=0.10 ;;
    4)  NAME="very_aggressive_enrich"; FRAC=0.15 ;;
    5)  NAME="both_large"; FRAC=0.15 ;;
    6)  NAME="both_large"; FRAC=0.25 ;;
    7)  NAME="multi_cpg_negatives"; FRAC=0.05 ;;
    8)  NAME="multi_cpg_negatives"; FRAC=0.10 ;;
    9)  NAME="kitchen_sink"; FRAC=0.10 ;;
    10) NAME="kitchen_sink"; FRAC=0.15 ;;
    11) NAME="kitchen_sink"; FRAC=0.20 ;;
    12) NAME="enrich_inactive_only"; FRAC=0.10 ;;
    13) NAME="enrich_inactive_only"; FRAC=0.15 ;;
    14) NAME="extreme_enrich"; FRAC=0.10 ;;
    15) NAME="extreme_enrich"; FRAC=0.15 ;;
    16) NAME="kitchen_sink"; FRAC=0.10
        DEBIAS_ARGS="++debias_mode=spectral ++debias_lambda=0.05" ;;
    17) NAME="kitchen_sink"; FRAC=0.10
        DEBIAS_ARGS="++debias_mode=cpg_invariance ++debias_lambda=1.0" ;;
    18) NAME="best_combo_scaled"; FRAC=0.15
        DEBIAS_ARGS="++debias_mode=cpg_invariance ++debias_lambda=0.5" ;;
    19) NAME="both_large"; FRAC=0.20
        DEBIAS_ARGS="++debias_mode=spectral ++debias_lambda=0.05" ;;
esac

# Build label
FRAC_INT=$(python3 -c "print(int(${FRAC}*100))")
LABEL="${NAME}_frac${FRAC_INT}"
if [ -n "${DEBIAS_ARGS}" ]; then
    MODE=$(echo "$DEBIAS_ARGS" | grep -oP 'debias_mode=\K\w+')
    LABEL="${LABEL}_${MODE}"
fi
OUT="outputs/oracle_neg_sweep/aggressive/${LABEL}/fold_0"

[ -f "${OUT}/test_metrics.json" ] && echo "SKIP: ${LABEL}" && exit 0

echo "=== ${LABEL} — $(date) ==="

CMD="uv run --no-sync python experiments/train_stage2_k562_hashfrag.py ${COMMON}"
CMD="${CMD} ++output_dir=${OUT}"
CMD="${CMD} ++negatives_path=${NEG_BASE}/${NAME}.tsv"
CMD="${CMD} ++neg_fraction=${FRAC}"
if [ -n "${DEBIAS_ARGS}" ]; then
    CMD="${CMD} ${DEBIAS_ARGS}"
fi

echo "CMD: ${CMD}"
eval ${CMD}

echo "=== ${LABEL} DONE — $(date) ==="
