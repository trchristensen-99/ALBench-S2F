#!/bin/bash
# Final push: aggressive CpG debiasing to reach random DNA target of +0.1-0.2.
#
# 5 strategies × multiple fractions + loss combos = 20 experiments
#
#  0: massive_combined 10%     (200K: enrich + deplete + multi-CpG negatives)
#  1: massive_combined 15%
#  2: massive_combined 20%
#  3: massive_combined 25%
#  4: heavy_neg_enrich 10%     (enrichment + 70K negatives + oversampled ctrl_neg)
#  5: heavy_neg_enrich 15%
#  6: heavy_neg_enrich 20%
#  7: heavy_neg_enrich 25%
#  8: pure_enrich_100k 20%    (pure enrichment, no negatives)
#  9: pure_enrich_100k 30%
# 10: graded_plus_neg 10%     (CpG ladder + tight negatives)
# 11: graded_plus_neg 15%
# 12: graded_plus_neg 20%
# 13: ctrl_neg_oversampled 10% (ctrl_neg at 5 CpG levels, oversampled)
# 14: ctrl_neg_oversampled 15%
# 15: ctrl_neg_oversampled 20%
# 16: massive_combined 15% + spectral λ=0.05
# 17: massive_combined 15% + cpg_invariance λ=1.0
# 18: heavy_neg_enrich 15% + cpg_invariance λ=1.0
# 19: graded_plus_neg 15% + group_dro λ=0.5
#
# Submit:
#   /cm/shared/apps/slurm/current/bin/sbatch --array=0-19 scripts/slurm/final_push_neg_aug.sh
#
#SBATCH --job-name=fin_neg
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
NEG_BASE="data/neg_aug_final_push"

COMMON="--config-name stage2_k562_oracle ++fold_id=0 ++n_folds=10 ++stage1_dir=${S1_DIR} ++use_full_dataset=True ++wandb_mode=offline"
DEBIAS_ARGS=""

case $T in
    0)  NAME="massive_combined"; FRAC=0.10 ;;
    1)  NAME="massive_combined"; FRAC=0.15 ;;
    2)  NAME="massive_combined"; FRAC=0.20 ;;
    3)  NAME="massive_combined"; FRAC=0.25 ;;
    4)  NAME="heavy_neg_enrich"; FRAC=0.10 ;;
    5)  NAME="heavy_neg_enrich"; FRAC=0.15 ;;
    6)  NAME="heavy_neg_enrich"; FRAC=0.20 ;;
    7)  NAME="heavy_neg_enrich"; FRAC=0.25 ;;
    8)  NAME="pure_enrich_100k"; FRAC=0.20 ;;
    9)  NAME="pure_enrich_100k"; FRAC=0.30 ;;
    10) NAME="graded_plus_neg"; FRAC=0.10 ;;
    11) NAME="graded_plus_neg"; FRAC=0.15 ;;
    12) NAME="graded_plus_neg"; FRAC=0.20 ;;
    13) NAME="ctrl_neg_oversampled"; FRAC=0.10 ;;
    14) NAME="ctrl_neg_oversampled"; FRAC=0.15 ;;
    15) NAME="ctrl_neg_oversampled"; FRAC=0.20 ;;
    16) NAME="massive_combined"; FRAC=0.15
        DEBIAS_ARGS="++debias_mode=spectral ++debias_lambda=0.05" ;;
    17) NAME="massive_combined"; FRAC=0.15
        DEBIAS_ARGS="++debias_mode=cpg_invariance ++debias_lambda=1.0" ;;
    18) NAME="heavy_neg_enrich"; FRAC=0.15
        DEBIAS_ARGS="++debias_mode=cpg_invariance ++debias_lambda=1.0" ;;
    19) NAME="graded_plus_neg"; FRAC=0.15
        DEBIAS_ARGS="++debias_mode=group_dro ++debias_lambda=0.5" ;;
esac

FRAC_INT=$(python3 -c "print(int(${FRAC}*100))")
LABEL="${NAME}_frac${FRAC_INT}"
if [ -n "${DEBIAS_ARGS}" ]; then
    MODE=$(echo "$DEBIAS_ARGS" | grep -oP 'debias_mode=\K\w+')
    LABEL="${LABEL}_${MODE}"
fi
OUT="outputs/oracle_neg_sweep/final_push/${LABEL}/fold_0"

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
