#!/bin/bash
# Comprehensive CpG debiasing combo experiments.
#
# Informed by finding that CpG-enriched augmentation (adding CpG to real
# training seqs, same labels) is most effective at breaking the CpG confound.
#
# Experiments:
#  0: cpg_enrich_aggressive (CpG+15, more aggressive) 5%
#  1: cpg_enrich_aggressive 10%
#  2: cpg_enrich_active_only (enrich only active seqs) 5%
#  3: cpg_enrich_active_only 10%
#  4: cpg_graded_ladder (0/0.03/0.06 per seq) 5%
#  5: cpg_graded_ladder 10%
#  6: enrich_plus_neg_combo (enrichment + random neg + dinuc neg) 5%
#  7: enrich_plus_neg_combo 10%
#  8: swap_plus_enrich (CG↔GC swap + enrichment) 5%
#  9: swap_plus_enrich 10%
# 10: jtt_cpg_upweight (upweight CpG-confound-breaking examples) 5%
# 11: jtt_cpg_upweight 10%
# 12: cpg_enrich_large (enrichment at 20% fraction)
# 13: ctrl_neg_cpg_enriched (ctrl_neg + CpG-enriched versions) 5%
# 14: cpg_enrich_aggressive + spectral loss (combined) 5%
# 15: cpg_enrich_aggressive + cpg_invariance loss (combined) 5%
# 16: enrich_plus_neg_combo + cpg_invariance loss (combined) 5%
# 17: cpg_graded_ladder + group_dro loss (combined) 5%
#
# Submit:
#   /cm/shared/apps/slurm/current/bin/sbatch --array=0-17 scripts/slurm/combo_neg_aug_batch.sh
#
#SBATCH --job-name=combo_neg
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
NEG_BASE="data/neg_aug_combos"

COMMON="--config-name stage2_k562_oracle ++fold_id=0 ++n_folds=10 ++stage1_dir=${S1_DIR} ++use_full_dataset=True ++wandb_mode=offline"

# Default: no loss debiasing
DEBIAS_ARGS=""

case $T in
    0)  NAME="cpg_enrich_aggressive"; FILE="cpg_enrich_aggressive.tsv"; FRAC=0.05 ;;
    1)  NAME="cpg_enrich_aggressive"; FILE="cpg_enrich_aggressive.tsv"; FRAC=0.10 ;;
    2)  NAME="cpg_enrich_active_only"; FILE="cpg_enrich_active_only.tsv"; FRAC=0.05 ;;
    3)  NAME="cpg_enrich_active_only"; FILE="cpg_enrich_active_only.tsv"; FRAC=0.10 ;;
    4)  NAME="cpg_graded_ladder"; FILE="cpg_graded_ladder.tsv"; FRAC=0.05 ;;
    5)  NAME="cpg_graded_ladder"; FILE="cpg_graded_ladder.tsv"; FRAC=0.10 ;;
    6)  NAME="enrich_plus_neg_combo"; FILE="enrich_plus_neg_combo.tsv"; FRAC=0.05 ;;
    7)  NAME="enrich_plus_neg_combo"; FILE="enrich_plus_neg_combo.tsv"; FRAC=0.10 ;;
    8)  NAME="swap_plus_enrich"; FILE="swap_plus_enrich.tsv"; FRAC=0.05 ;;
    9)  NAME="swap_plus_enrich"; FILE="swap_plus_enrich.tsv"; FRAC=0.10 ;;
    10) NAME="jtt_cpg_upweight"; FILE="jtt_cpg_upweight.tsv"; FRAC=0.05 ;;
    11) NAME="jtt_cpg_upweight"; FILE="jtt_cpg_upweight.tsv"; FRAC=0.10 ;;
    12) NAME="cpg_enrich_large"; FILE="cpg_enrich_large.tsv"; FRAC=0.20 ;;
    13) NAME="ctrl_neg_cpg_enriched"; FILE="ctrl_neg_cpg_enriched.tsv"; FRAC=0.05 ;;
    # Combined: data + loss approaches
    14) NAME="enrich_agg_spectral"; FILE="cpg_enrich_aggressive.tsv"; FRAC=0.05
        DEBIAS_ARGS="++debias_mode=spectral ++debias_lambda=0.05" ;;
    15) NAME="enrich_agg_cpginv"; FILE="cpg_enrich_aggressive.tsv"; FRAC=0.05
        DEBIAS_ARGS="++debias_mode=cpg_invariance ++debias_lambda=1.0" ;;
    16) NAME="combo_cpginv"; FILE="enrich_plus_neg_combo.tsv"; FRAC=0.05
        DEBIAS_ARGS="++debias_mode=cpg_invariance ++debias_lambda=1.0" ;;
    17) NAME="ladder_gdro"; FILE="cpg_graded_ladder.tsv"; FRAC=0.05
        DEBIAS_ARGS="++debias_mode=group_dro ++debias_lambda=0.5" ;;
esac

FRAC_PCT=$(echo "$FRAC" | sed 's/0\.\(.*\)/\1/' | sed 's/^0//')
LABEL="${NAME}_frac${FRAC_PCT}"
OUT="outputs/oracle_neg_sweep/combo_batch/${LABEL}/fold_0"

[ -f "${OUT}/test_metrics.json" ] && echo "SKIP: ${LABEL}" && exit 0

echo "=== ${LABEL} — $(date) ==="

CMD="uv run --no-sync python experiments/train_stage2_k562_hashfrag.py ${COMMON}"
CMD="${CMD} ++output_dir=${OUT}"
CMD="${CMD} ++negatives_path=${NEG_BASE}/${FILE}"
CMD="${CMD} ++neg_fraction=${FRAC}"
if [ -n "${DEBIAS_ARGS}" ]; then
    CMD="${CMD} ${DEBIAS_ARGS}"
fi

echo "CMD: ${CMD}"
eval ${CMD}

echo "=== ${LABEL} DONE — $(date) ==="
