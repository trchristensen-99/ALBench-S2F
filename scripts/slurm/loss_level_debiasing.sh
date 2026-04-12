#!/bin/bash
# Loss-level debiasing experiments.
#
# Three approaches that modify the training loss to remove CpG confound:
#
# 1. Spectral decoupling: L2 penalty on predictions (pred^2)
#    Slows learning of "easy" features (CpG shortcut), forcing harder features
#
# 2. Group DRO: worst-group optimization across CpG × activity bins
#    Forces model to perform well on hard groups (high-CpG inactive)
#
# 3. CpG invariance: penalty on correlation between residuals and CpG content
#    Directly penalizes CpG-dependent prediction errors
#
# Each tested at multiple lambda values, with and without neg-aug:
#
#   0: spectral, lambda=0.01
#   1: spectral, lambda=0.05
#   2: spectral, lambda=0.10
#   3: group_dro, lambda=0.3
#   4: group_dro, lambda=0.5
#   5: group_dro, lambda=0.7
#   6: cpg_invariance, lambda=0.5
#   7: cpg_invariance, lambda=1.0
#   8: cpg_invariance, lambda=2.0
#   9: spectral + counterfactual aug (combined)
#  10: group_dro + counterfactual aug (combined)
#  11: cpg_invariance + counterfactual aug (combined)
#
# Submit:
#   /cm/shared/apps/slurm/current/bin/sbatch --array=0-11 scripts/slurm/loss_level_debiasing.sh
#
#SBATCH --job-name=loss_deb
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
    # Pure loss modifications (no neg-aug data)
    0) MODE=spectral;       LAM=0.01; NEG_ARGS="" ;;
    1) MODE=spectral;       LAM=0.05; NEG_ARGS="" ;;
    2) MODE=spectral;       LAM=0.10; NEG_ARGS="" ;;
    3) MODE=group_dro;      LAM=0.3;  NEG_ARGS="" ;;
    4) MODE=group_dro;      LAM=0.5;  NEG_ARGS="" ;;
    5) MODE=group_dro;      LAM=0.7;  NEG_ARGS="" ;;
    6) MODE=cpg_invariance; LAM=0.5;  NEG_ARGS="" ;;
    7) MODE=cpg_invariance; LAM=1.0;  NEG_ARGS="" ;;
    8) MODE=cpg_invariance; LAM=2.0;  NEG_ARGS="" ;;
    # Combined: loss modification + counterfactual CpG augmentation
    9)  MODE=spectral;       LAM=0.05; NEG_ARGS="++negatives_path=data/cpg_counterfactual/cpg_bidirectional_swap.tsv ++neg_fraction=0.05" ;;
    10) MODE=group_dro;      LAM=0.5;  NEG_ARGS="++negatives_path=data/cpg_counterfactual/cpg_bidirectional_swap.tsv ++neg_fraction=0.05" ;;
    11) MODE=cpg_invariance; LAM=1.0;  NEG_ARGS="++negatives_path=data/cpg_counterfactual/cpg_bidirectional_swap.tsv ++neg_fraction=0.05" ;;
esac

LABEL="${MODE}_lam${LAM}"
if [ -n "${NEG_ARGS}" ]; then
    LABEL="${LABEL}_cf"
fi
OUT="outputs/oracle_neg_sweep/loss_debiasing/${LABEL}/fold_0"

[ -f "${OUT}/test_metrics.json" ] && echo "SKIP: ${LABEL}" && exit 0

echo "=== ${LABEL} — $(date) ==="

CMD="uv run --no-sync python experiments/train_stage2_k562_hashfrag.py ${COMMON}"
CMD="${CMD} ++output_dir=${OUT}"
CMD="${CMD} ++debias_mode=${MODE} ++debias_lambda=${LAM}"
if [ -n "${NEG_ARGS}" ]; then
    CMD="${CMD} ${NEG_ARGS}"
fi

echo "CMD: ${CMD}"
eval ${CMD}

echo "=== ${LABEL} DONE — $(date) ==="
