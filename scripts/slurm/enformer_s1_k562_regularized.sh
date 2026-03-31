#!/bin/bash
# Enformer S1 K562 with stronger regularization for better OOD generalization.
#
# The default Enformer S1 K562 overfits (27 epochs, val=0.872 but OOD=0.20-0.32).
# Other models find K562 OOD the easiest cell line, so the issue is head overfitting.
# S2 fine-tuning fixes it (OOD=0.58-0.63), but for fair S1 comparison we want
# a well-regularized S1 head.
#
# Sweep: 3 configs × 3 seeds = 9 tasks
#   0-2: dropout=0.3, wd=1e-4, patience=5, lr=0.001  (best from grid search)
#   3-5: dropout=0.5, wd=1e-3, patience=3, lr=0.001  (aggressive regularization)
#   6-8: dropout=0.3, wd=1e-4, patience=3, lr=0.002  (higher LR + early stop)
#
#SBATCH --job-name=enf_s1_reg
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

set -euo pipefail

set +u; source /etc/profile.d/modules.sh; set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
source scripts/slurm/setup_hpc_deps.sh

T=$SLURM_ARRAY_TASK_ID
echo "=== Enformer S1 K562 regularized task=${T} node=${SLURMD_NODENAME} date=$(date) ==="

SEEDS=(42 123 456)
SEED_IDX=$((T % 3))
SEED="${SEEDS[$SEED_IDX]}"
CONFIG_IDX=$((T / 3))

case ${CONFIG_IDX} in
0)
    LABEL="do0.3_wd1e-4_pat5_lr1e-3"
    uv run --no-sync python experiments/train_foundation_cached.py \
        ++model_name=enformer \
        ++cell_line=k562 \
        ++cache_dir=outputs/enformer_k562_cached/embedding_cache \
        ++embed_dim=3072 \
        ++output_dir="outputs/enformer_k562_regularized/${LABEL}/seed_${SEED}" \
        ++seed=${SEED} \
        ++lr=0.001 ++weight_decay=0.0001 ++dropout=0.3 \
        ++epochs=100 ++early_stop_patience=5
    ;;
1)
    LABEL="do0.5_wd1e-3_pat3_lr1e-3"
    uv run --no-sync python experiments/train_foundation_cached.py \
        ++model_name=enformer \
        ++cell_line=k562 \
        ++cache_dir=outputs/enformer_k562_cached/embedding_cache \
        ++embed_dim=3072 \
        ++output_dir="outputs/enformer_k562_regularized/${LABEL}/seed_${SEED}" \
        ++seed=${SEED} \
        ++lr=0.001 ++weight_decay=0.001 ++dropout=0.5 \
        ++epochs=100 ++early_stop_patience=3
    ;;
2)
    LABEL="do0.3_wd1e-4_pat3_lr2e-3"
    uv run --no-sync python experiments/train_foundation_cached.py \
        ++model_name=enformer \
        ++cell_line=k562 \
        ++cache_dir=outputs/enformer_k562_cached/embedding_cache \
        ++embed_dim=3072 \
        ++output_dir="outputs/enformer_k562_regularized/${LABEL}/seed_${SEED}" \
        ++seed=${SEED} \
        ++lr=0.002 ++weight_decay=0.0001 ++dropout=0.3 \
        ++epochs=100 ++early_stop_patience=3
    ;;
esac

echo "=== Done: $(date) ==="
